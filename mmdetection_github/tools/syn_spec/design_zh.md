# RAW 图像仿真流水线

## 概述

给定一张源 RAW 图像（由单一相机拍摄），该流水线模拟同一场景在不同成像条件下会呈现出的样子：包括不同的相机光谱敏感度、不同的光源、不同的曝光，以及不同的传感器特性。其目标是为检测任务提供数据增强。

所有操作都在**线性 RAW 空间**中完成（完成 black level 减除之后）。输出是一个位于 `[0, 1]` 范围内的模拟线性 RAW 图像。

---

## 输入

- `raw`：源 RAW 图像，形状为 `(H, W, 3)`，线性、已减 black level，并归一化到 `[0, 1]`
- `camspec_database.txt`：来自 Jiang et al. (2013) 的 28 台相机光谱敏感度数据库

---

## 文件 `camspec_database.txt` 的格式

该文件包含 28 台相机的光谱敏感度数据。每台相机占据**连续的 3 行**，分别对应 R、G、B 三个通道。每一行包含 **33 个数值**，对应 400 nm 到 720 nm、步长为 10 nm 的波长采样点。

```python
# 总行数：28 台相机 × 3 个通道 = 84 行
# 每行：33 个浮点数（对应 400, 410, ..., 720 nm 处的敏感度）
# 每台相机内部的行顺序：R 通道、G 通道、B 通道
```

加载示例：

```python
import numpy as np

raw_data = np.loadtxt('camspec_database.txt')  # 形状: (84, 33)
cameras = raw_data.reshape(28, 3, 33)          # 形状: (28, 3, 33)
# cameras[i, 0, :] -> 第 i 台相机的 R 通道，33 个波长点
# cameras[i, 1, :] -> 第 i 台相机的 G 通道
# cameras[i, 2, :] -> 第 i 台相机的 B 通道
# 转置成 (28, 33, 3) 以便后续使用：
cameras = cameras.transpose(0, 2, 1)           # 形状: (28, 33, 3)
```

每一行都经过了**峰值归一化**（每台相机每个通道的最大值为 1）。

---

## 离线预计算（在增强开始前仅运行一次）

### Step O-1：加载并归一化相机数据库

```python
cameras = load_camspec(path)       # 形状: (28, 33, 3)
wavelengths = np.arange(400, 730, 10)  # 33 个采样点，单位 nm
```

### Step O-2：对每个通道分别拟合 PCA

使用 `sklearn.decomposition.PCA`，并设 `n_components=2`（Jiang et al. 表明 2 个主成分可以解释超过 97% 的方差）。

```python
from sklearn.decomposition import PCA

pca = {}          # 字典：通道名 -> 拟合好的 PCA 对象
coeffs_all = {}   # 字典：通道名 -> (28, 2) 的 PCA 系数数组

for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
    X = cameras[:, :, ch_idx]          # (28, 33)
    p = PCA(n_components=2)
    coeffs = p.fit_transform(X)        # (28, 2)
    pca[ch_name] = p
    coeffs_all[ch_name] = coeffs
```

### Step O-3：根据数据计算采样统计量

```python
stats = {}
for ch_name in ['R', 'G', 'B']:
    c = coeffs_all[ch_name]           # (28, 2)
    stats[ch_name] = {
        'mean': c.mean(axis=0),       # 形状 (2,)
        'std':  c.std(axis=0),        # 形状 (2,)
        'min':  c.min(axis=0),        # 形状 (2,) —— 用于裁剪
        'max':  c.max(axis=0),        # 形状 (2,) —— 用于裁剪
    }
```

这 28 台真实相机在 PCA 矩形空间中并不是均匀分布的，而是集中在中心附近。采用**高斯采样** `N(mean, std²)`（再裁剪到 `[min, max]`）可以生成更多集中于典型真实相机附近、同时仍覆盖整体范围的目标相机。相比之下，均匀采样会过度采样极端角落，而这些角落可能对应物理上并不合理的光谱敏感度。

### Step O-4：为源相机预计算 ColorChecker 响应（使用 CIE CMF 近似）

由于源相机真实的 `S_c^A(λ)` 未知，这里使用数据库中 **28 台相机的平均值** 来作为源相机的代理。相比直接使用 CIE 1931 CMF，这样更合适，因为 CIE 的 Z 函数峰值约为 1.77，远高于真实相机蓝通道通常约 0.8 的峰值；若直接用 CIE CMF，会压缩交叉相机矩阵中的蓝通道动态范围，并使冷色光看起来过于中性。

```python
# 在 400-720 nm、10 nm 采样点上加载 CIE 1931 2° 标准观察者 CMF
# 形状: (33, 3)，三列分别是 xbar, ybar, zbar
cie_cmf = load_cie_cmf(wavelengths)

# 加载 Macbeth ColorChecker 24 色块的反射率光谱
# 形状: (33, 24)，每一列是一个色块在 400-720 nm 上的反射率
cc_reflectance = load_colorchecker(wavelengths)

# D65 光源光谱，形状 (33,)
d65 = load_D65(wavelengths)

# 源相机代理：数据库中 28 台相机的平均
source_cam = cameras.mean(axis=0)   # (33, 3)

# 源相机在 D65 光照下对 ColorChecker 的响应
# P_A 形状: (24, 3)
P_A = (cc_reflectance * d65[:, None]).T @ source_cam * 10.0   # (24, 3)
P_A /= P_A[:, 1:2]   # 归一化，使 G 通道 = 1

# 预计算标准化后的 D65 光谱，用作色温放大时的参考
I_d65 = d65 / (d65.max() + 1e-8)
```

---

## 在线增强（对每一张图像执行）

下面的步骤会在训练时对每张图像独立执行。

---

### Step 0：自动白平衡归一化

未做白平衡的 RAW 图像通常存在明显的绿色偏置（Bayer 模式是 2G 对 1R 1B，通常 G/R 约为 2.0）。而交叉相机矩阵默认假设输入通道是大致平衡的。因此，在变换前先做一次对角白平衡归一化，并在变换后**再撤销这一步**，这样输出仍然保留 RAW 图像典型的绿色优势。

```python
ch_means = raw.mean(axis=(0, 1)) + 1e-8       # (3,)
wb_gain = ch_means[1] / ch_means               # 以 G 通道为基准归一化
raw_balanced = raw * wb_gain[None, None, :]     # 此时各通道均值大致接近
```

---

### Step 1：采样曝光

在对数空间（stops）中采样全局曝光增益 `α`，并且它与其他所有参数**相互独立**：

$$u \sim \mathcal{U}(u_{\min},\, u_{\max}), \qquad \alpha = 2^u$$

```python
u_min, u_max = -3.0, 3.0        # 总共覆盖 6 档曝光范围
u = np.random.uniform(u_min, u_max)
alpha = 2.0 ** u
```

---

### Step 2：采样色温（与曝光独立）

在 **mired** 尺度上采样色温，其中 `m = 10^6 / T`。之所以采用 mired，是因为在 Planckian locus 上，相等的 mired 间隔大致对应相等的感知色度变化。

在 mired 空间中做均匀采样：

$$m \sim \mathcal{U}(m_{\min},\, m_{\max})$$

```python
m_min = 50    # 对应 T = 20000 K（极冷）
m_max = 400   # 对应 T = 2500 K（钨丝灯/烛光）

m = np.random.uniform(m_min, m_max)
T = 1e6 / m   # 色温，单位 Kelvin
```

---

### Step 3：采样 tint（偏离 Planckian locus 的分量）

真实光源（如荧光灯、LED）并不严格落在理想的 Planckian locus 上。这里用一个扰动项 `Δuv` 来描述这种偏离，它控制的是 **绿-洋红轴**（与暖-冷轴正交）：

$$\Delta uv \sim \mathcal{U}(-0.04,\, 0.04)$$

正的 `Δuv` 表示偏绿，负的 `Δuv` 表示偏洋红。

```python
delta_uv = np.random.uniform(-0.04, 0.04)
```

这里并不直接通过修改光谱来施加 tint，而是将其实现为 **RGB 空间中的对角增益**，因为直接通过 CMF 伪逆去修改光谱在数值上很不稳定。对于偏绿 tint，会同步降低 R 和 B，其中 R 的变化幅度是 B 的一半，以避免出现过强的偏红：

```python
def tint_to_diagonal(delta_uv):
    strength = 15.0
    r_ratio = 0.5   # 为避免红色过强，R 的响应强度只有 B 的一半
    gain_r = max(1.0 - delta_uv * strength * r_ratio, 0.1)
    gain_b = max(1.0 - delta_uv * strength, 0.1)
    return np.array([gain_r, 1.0, gain_b])

tint_gain = tint_to_diagonal(delta_uv)   # 形状: (3,)
```

---

### Step 4：计算黑体光源光谱 `I_T(λ)`

使用 Planck 定律来计算采样得到的光源的光谱功率分布：

$$B(\lambda, T) = \frac{2\pi h c^2}{\lambda^5 \left(\exp\!\left(\frac{hc}{kT\lambda}\right) - 1\right)}$$

```python
def planckian_spectrum(T, wavelengths_nm):
    h = 6.62607e-34   # 普朗克常数, J·s
    c = 2.99792e8     # 光速, m/s
    k = 1.38065e-23   # 玻尔兹曼常数, J/K
    lam = wavelengths_nm * 1e-9   # nm 转 m
    B = (2 * np.pi * h * c**2) / (lam**5 * (np.expm1(h * c / (k * T * lam))))
    return B / B.max()   # 峰值归一化为 1

I_T = planckian_spectrum(T, wavelengths)   # 形状: (33,)
```

---

### Step 5：tint 在 Step 8 中施加

tint 过去曾尝试通过 CMF 伪逆来直接修改光源光谱实现，但这种方式数值不稳定（33×3 矩阵的伪逆会产生接近 0 甚至为负的光谱）。因此现在改为在 Step 3 中将 tint 表示为 RGB 对角增益，并在 Step 8 中施加到交叉相机矩阵上。

---

### Step 6：采样目标相机的光谱敏感度 `S_c^B(λ)`

在 Step O-3 得到的 28 台真实相机的 PCA 统计量基础上，采用**高斯分布**进行采样，再裁剪到 `[min, max]`，以保证落在物理合理的范围内：

```python
def sample_spectral_sensitivity(pca, stats, wavelengths):
    S = np.zeros((len(wavelengths), 3))
    for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
        s = stats[ch_name]
        sigma = np.random.normal(s['mean'], s['std'])   # 形状 (2,)
        sigma = np.clip(sigma, s['min'], s['max'])       # 裁剪到有效范围
        curve = pca[ch_name].inverse_transform(sigma[None])[0]   # (33,)
        curve = np.clip(curve, 0.0, None)   # 强制非负
        curve /= (curve.max() + 1e-8)       # 峰值归一化为 1
        S[:, ch_idx] = curve
    return S   # 形状: (33, 3)

S_B = sample_spectral_sensitivity(pca, stats, wavelengths)
```

---

### Step 7：计算交叉相机颜色变换矩阵 `M_{A → B, T}`

该矩阵同时编码了 **相机光谱敏感度变化**（`S_c^A → S_c^B`）和 **光源变化**（`D65 → I_T`）。其计算方法是：在两种条件下分别计算 ColorChecker 的响应，然后用最小二乘拟合一个线性映射。tint 在 Step 8 中单独处理。

```python
def compute_cross_camera_matrix(P_A, S_B, I_T, cc_reflectance, d_lambda=10):
    # 目标相机在光源 I_T 下对 ColorChecker 的响应
    P_B = (cc_reflectance * I_T[:, None]).T @ S_B * d_lambda   # (24, 3)
    P_B /= P_B[:, 1:2] + 1e-8   # 归一化，使 G = 1

    # 最小二乘：求 M (3x3)，使得 P_A @ M^T ≈ P_B
    M, _, _, _ = np.linalg.lstsq(P_A, P_B, rcond=None)
    return M.T   # 形状: (3, 3)

M_cross = compute_cross_camera_matrix(P_A, S_B, I_T, cc_reflectance)
```

---

### Step 8：采样传感器噪声参数，并组装最终变换矩阵

```python
# 非对角通道混合（模拟传感器光谱串扰）
epsilon = np.random.uniform(-0.05, 0.05, size=(3, 3))
np.fill_diagonal(epsilon, 0.0)

# black level 偏移（每个通道的 pedestal 变化）
delta = 0.02
beta = np.random.uniform(-delta, delta, size=3)

# 将 tint 作为对角增益作用到交叉相机矩阵上
# tint_gain 来自 Step 3: [gain_r, 1.0, gain_b]
M_tinted = np.diag(tint_gain) @ M_cross

# 最终矩阵：曝光 × (带 tint 的交叉相机矩阵 + 串扰项)
M_full = alpha * (M_tinted + epsilon)   # 形状: (3, 3)
```

---

### Step 9：应用变换，并恢复 RAW 的绿色偏置

```python
H, W = raw_balanced.shape[:2]
v_flat = raw_balanced.reshape(-1, 3)                  # (N, 3)
v_out = (M_full @ v_flat.T).T + beta[None, :]         # (N, 3)
v_out = v_out.reshape(H, W, 3)

# 撤销自动白平衡，以恢复 RAW 图像典型的绿色优势
# 真实 RAW 图像通常满足 G > R, G > B，这是 Bayer 模式导致的
v_out = v_out / wb_gain[None, None, :]
```

---

### Step 10：高光 roll-off 与饱和裁剪

使用一个 soft-clip（`tanh`）来模拟传感器接近 full-well capacity 时的非线性，再接一个硬裁剪：

$$f_\gamma(v) = S \cdot \tanh\!\left(\frac{v}{S}\right)$$

```python
S_sat = np.random.uniform(0.85, 1.0)   # 随机饱和点

v_rolled = S_sat * np.tanh(v_out / S_sat)
v_clipped = np.clip(v_rolled, 0.0, S_sat)
```

---

### Step 11：量化

量化到目标位深 `b`：

$$\tilde{v} = \left\lfloor \frac{v'}{S} \cdot (2^b - 1) + 0.5 \right\rfloor / (2^b - 1)$$

```python
b = np.random.choice([10, 12, 14, 16])
v_quant = np.floor(v_clipped / S_sat * (2**b - 1) + 0.5) / (2**b - 1)
```

最终输出 `v_quant` 就是位于 `[0, 1]` 范围内的模拟 RAW 图像。

---

## 完整数据流

```text
raw ∈ [0,1]  （线性、已减 black level、带绿色偏置）
    │
    ├── Step 0:  auto_wb: raw_bal = raw × wb_gain               [归一化绿色偏置]
    │
    ├── Step 1:  采样 α = 2^u,  u ~ U(-3, 3)                    [曝光]
    ├── Step 2:  采样 m ~ U(50, 400),  T = 1e6/m                [色温, mired]
    ├── Step 3:  采样 Δuv ~ U(-0.04, 0.04) → tint_gain          [tint, RGB 对角增益]
    │
    ├── Step 4:  B(λ, T) → I_T(λ)                               [黑体光谱]
    ├── Step 5:  （tint 在 Step 8 中施加）
    ├── Step 6:  PCA N(μ,σ²) 采样 → S_c^B(λ)                    [目标相机]
    │
    ├── Step 7:  ColorChecker bridge (P_A, P_B) → M_cross       [交叉相机 + 光源变换]
    ├── Step 8:  M_full = α·(diag(tint) @ M_cross + ε)          [组装最终矩阵]
    │
    ├── Step 9:  v' = M_full @ raw_bal + β                      [应用变换]
    │            v' = v' / wb_gain                              [恢复绿色偏置]
    ├── Step 10: v'' = S·tanh(v'/S), clip 到 [0, S]             [高光 roll-off]
    └── Step 11: 量化到 b-bit                                    [量化]
            │
            └── simulated RAW ∈ [0,1]  （保留绿色偏置，符合 RAW 特性）
```

---

## 所需 Python 库

```python
numpy
scipy
scikit-learn
colour-science   # 用于 CIE CMF、D65 光源、ColorChecker 反射率
```

用 `colour-science` 加载 CIE 数据的示例：

```python
import colour

# 自定义波长下的 CIE 1931 2° 标准观察者 CMF
cmf_obj = colour.colorimetry.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
cie_cmf = colour.colorimetry.msds_to_XYZ(
    cmf_obj, wavelengths=wavelengths
)   # 也可以选择手动插值

# D65 光源
sd_D65 = colour.SDS_ILLUMINANTS['D65']
d65 = sd_D65.values   # 需要插值到你的波长点

# Macbeth ColorChecker 反射率
cc = colour.characterisation.SDS_COLOURCHECKERS['ColorChecker 2005']
cc_reflectance = np.stack([
    sds.values for sds in cc.values()
], axis=1)   # 同样需要插值到你的波长点
```

另一种方式是从 `cvrl.org` 下载 CIE CMF 和 D65 的 CSV 文件。

---

## 超参数汇总

| 参数 | 符号 | 数值 | 说明 |
|---|---|---|---|
| 曝光范围 | `u` | `U(-3, 3)` stops | 共 6 档范围 |
| mired 范围 | `m` | `U(50, 400)` | 对应 2500 K – 20000 K |
| Tint | `Δuv` | `U(-0.04, 0.04)` | 均匀采样；绿-洋红轴 |
| Tint 强度 | — | `15.0` | 每单位 `Δuv` 对 R/B 的增益变化 |
| Tint 的 R 比例 | — | `0.5` | R 的响应强度是 B 的一半，避免过红 |
| 串扰项 | `ε_ij` | `U(-0.05, 0.05)` | 仅非对角项，并随曝光一起缩放 |
| Black level 偏移 | `β_c` | `U(-0.02, 0.02)` | 每通道独立 |
| 饱和点 | `S` | `U(0.85, 1.0)` | |
| 位深 | `b` | `{10, 12, 14, 16}` | 均匀选择 |
| PCA 主成分数 | `k` | `2` | 每个通道单独做 PCA，依据 Jiang et al. 2013 |
| PCA 采样方式 | — | `N(μ, σ²)`，并裁剪到 `[min, max]` | 从 28 台相机拟合得到 |
| 源相机 | — | 28 台相机均值 | 相比 CIE CMF 更能保留蓝通道动态范围 |
| 自动白平衡 | — | 变换后再撤销 | 用于保留 RAW 的绿色偏置 |

---

## 实现说明

1. **边界框标签不受影响**：所有操作都是逐像素的颜色变换，没有任何空间形变，因此检测框无需修改。

2. **批处理效率**：Step O-1 到 O-4（PCA 拟合、ColorChecker 预计算）应在启动时做一次并缓存。真正对每张图执行的只有 Step 0 到 Step 11。

3. **PCA 系数采样**：采用从 28 台真实相机中拟合得到的高斯采样 `N(mean, std²)`，再裁剪到 `[min, max]`，从而保证落在物理合理的凸包内。这使得采样结果主要集中在典型相机附近，但尾部仍可覆盖较极端情况。

4. **`camspec_database.txt` 的行顺序**：需要确认文件中确实是按每台相机的 `R/G/B` 顺序排列。可以通过检查每行是否非负、峰值是否接近 1 来验证。如果文件采用其他排列方式（例如先全部 R，再全部 G，再全部 B），则需要相应调整 reshape 逻辑。

5. **源相机代理**：这里采用 28 台相机的均值，而不是直接使用 CIE 1931 CMF。原因是 CIE 的 Z 函数（蓝通道）峰值约为 1.77，明显高于真实相机蓝通道的典型峰值（约 0.8），会压缩交叉相机矩阵中的蓝通道动态范围。

6. **Tint 的实现方式**：tint 作为 RGB 对角增益来施加，而不是通过修改光源光谱实现。基于 CMF 伪逆的方法在数值上不稳定，即使 `tint=0`，33×3 伪逆也可能产生接近 0 或负值的光谱。对角增益实现简单、稳定，而且视觉效果合理。

7. **自动白平衡的撤销**：输出必须保留 RAW 图像典型的绿色优势（`G > R, G > B`），因为下游检测模型就是在这种绿色偏置的 RAW 输入上训练的。如果不撤销，输出会更像一张带色偏的白平衡图，而不是 RAW。

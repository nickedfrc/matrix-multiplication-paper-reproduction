# 论文复现与矩阵乘法对比

本目录对应用户提供的论文 **An Upper Bound of 9/4 for the Matrix Multiplication Exponent**（OpenAI，2026-10-02）。
官方来源：https://github.com/openai/math/tree/main/preprints/Matrix-Multiplication-Nine-Fourths-October-2-2026

## 核心结论和实现边界

论文证明复数算术模型下 ω ≤ 9/4，即对于每个 ε > 0，存在 Oε(n^(9/4+ε)) 算术操作的算法。
**这不等于给出了一套能直接运行的 n^2.25 矩阵乘法公式。**

第 9 页从指数的定义推出：存在固定整数 u ≥ 2，以及矩阵乘法张量 T_u 的长度为 r 的精确秩分解，
其中 r < u^(9/4+δ)。随后对块矩阵递归，得到：

    S(N) ≤ r S(N/u) + C_(u,r) (N/u)^2

实际实现必须取得 u、r 和每项的三组系数 U、V、W。论文没有列出这些数据，也没有给出将证明转成这些
数据的可执行构造或有效规模界限。因此，本项目没有声称实现或测量论文的完整 9/4 算法。
缺少的是具体秩分解，不能通过把 Strassen 换一个名称来补足。

## 文件

- matmul_compare.py：朴素 Python 三重循环、NumPy 外积累加、分块 NumPy、Strassen、NumPy @；正确性与计时。
- paper_constructions.py：论文中明确可计算的有限张量构造。
- bilinear_scheme.py：给定 U、V、W 后的通用递归执行器；支持 .npz 系数文件，用 Strassen 验证执行器。
- results/benchmark.md：完整实测表格。
- results/benchmark.json：逐次计时、误差、环境、输入与线程配置。
- results/paper_checks.json：论文有限构造的验证结果。

## 已复现的论文内容

| 论文位置 | 代码 | 验证 |
|---|---|---|
| 命题 3.1，第 4–5 页 | separation | 5M 份 Fourier 投影、共享腿分离、平方权重与 ε→0 极限 |
| 引理 4.1，第 7 页 | determinant_filtration | 所有第一输入切片同时保持滤过；两个对角块分别为 C(a,b+1)、C(a,b-1) |
| 引理 4.2，第 8 页 | three_sectors | 权重非负；左右块为 C(a,h)，中块为交换两腿并反转第一腿后的 C(a,h) |
| 引理 5.1，第 9 页 | diagonal_growth | H 的有限乘积和 D_a ≥ a^(4/3) 下界的数值检查 |
| 定理 1.1 的递归步骤，第 9 页 | BilinearScheme | 有具体秩分解时，按线性组合、递归乘法、输出重组执行 |

Fourier 选择器先用复根求和进行数值核对，再用整数等式精确保留项；ε=0 时保留权重零项，
避免负权重和浮点抵消造成错误。张量存储省略了必定匹配的第二份扇区标签，因此使用压缩的 7 维系数数组。

这些有限实验不计算抽象张量特征 λ 或谱剖面 P，也不验证整个非构造性存在证明。
测试通过意味着代码对应的有限恒等式成立，不是完整定理的独立证明。

## 实测结果

2026-10-09，本机 Windows，Python 3.12.14，NumPy 2.3.5，构建后端 OpenBLAS 0.3.30。
float64 方阵、固定随机种子 20261009、5 次测量取中位数，耗时单位 ms。

| 阶数 | NumPy @ | 分块 128 | Strassen 叶阈值 256 | NumPy 外积累加 |
|---|---:|---:|---:|---:|
| 512 | 6.3465 | 9.9175 | 10.3521 | 433.8910 |
| 1024 | 44.1958 | 75.6599 | 97.8113 | 4586.9963 |

128 阶的朴素三重循环为 367.3001 ms，NumPy 为 0.1037 ms。
在测试的阈值 32、64、128、256 中，1024 阶最好的 Strassen 是阈值 256，
耗时约为 NumPy 的 2.21 倍。这个结论仅适用于本次机器、实现和参数范围。

所有 Strassen 计时均至少发生一层递归；叶阈值不小于矩阵大小的情形被排除。
计时包含输入检查、转换、分配和必要补零，排除数据生成和正确性比较。
BLAS 环境变量在导入 NumPy 之前设为 1。独立新进程查询 OpenBLAS 返回线程数 1；原始计时进程仅记录了环境设置，未直接读取线程计数。
小矩阵使用批量调用减少计时噪声；方法顺序打乱，先预热，再保存原始样本。
每个实现使用同一组输入。NumPy 作为浮点参考，不是独立高精度真值。

286 项矩阵乘法检查通过，覆盖基矩阵对、实际多层递归、非规则尺寸、空维度、
复数、非连续视图、缩放和非法形状。性能测试的每个结果也通过误差检查。
1024 阶所测 Strassen 相对 Frobenius 误差最大约 6.73e-15。
有限构造测试：6 个 Fourier 分离实例、25 个行列式滤过实例、30 个三扇区实例；
Fourier 投影最大数值误差约 1.22e-15。

## 运行

在本目录运行，依赖仅 NumPy：

    python -m pip install -r requirements.txt
    python paper_constructions.py
    python bilinear_scheme.py
    python matmul_compare.py
    python matmul_compare.py --verify-only
    python matmul_compare.py --sizes 256 512 1024 2048 --repeats 7 --naive-max 64

## 给定系数后如何运行递归算法

系数定义：

    L_s = Σ_ab U[s,a,b] A_ab
    R_s = Σ_cd V[s,c,d] B_cd
    C_ij = Σ_s W[i,j,s] (L_s @ R_s)

U,V 形状为 (r,u,u)，W 形状为 (u,u,r)。递归保持左输入在右输入之前，不交换矩阵乘法次序。
示例：

    import numpy as np
    from bilinear_scheme import BilinearScheme, strassen_certificate

    scheme = strassen_certificate()  # 经典 Strassen，u=2,r=7，指数约 2.80735
    scheme.verify()
    a = np.arange(25).reshape(5,5)
    b = np.eye(5)
    c = scheme.multiply(a,b,leaf_size=1)
    np.testing.assert_allclose(c,a@b)

    # 获得真正的论文秩分解后：
    # np.savez("rank_certificate.npz", U=U, V=V, W=W)
    # scheme = BilinearScheme.load("rank_certificate.npz")
    # c = scheme.multiply(a,b,leaf_size=128)

verify 穷举基矩阵对检查双线性恒等式；对一般浮点系数它只是数值检查，
不能代替代数数或有理数系数的精确证明。执行器使用 complex128，支持同尺寸方阵；
Strassen 专用实现还支持长方形矩阵和非 2 的幂尺寸。

所有输入都转换为至少 float64，复数通常为 complex128；不适用于大整数精确计算。
任意形状 Strassen 会补成方阵，细长矩阵可能产生很大内存开销。
有限墙钟计时不能验证渐近指数，且论文的复数算术操作模型不包含位复杂度、缓存和硬件常数。


## 来源、许可证与 AI 辅助声明

本项目由维护者使用 **OpenAI Codex 辅助编写、整理文档并运行测试**。
数学成果归所引用论文作者；本仓库未提出新的矩阵乘法指数结果，亦非 OpenAI 官方实现。
代码在本次开发中依据论文数学定义独立编写，未复制上游 Python 或 Lean 源代码。
仓库不包含论文 PDF、正文提取、论文插图或知乎回答；仅提供论文链接及必要的数学引用。

仓库的原创代码和文档采用 Apache License 2.0，完整条款见 LICENSE。
论文和上游材料的权利归原作者；本许可证不为第三方内容重新授权。
NumPy 是外部依赖，不随仓库分发，其许可证由 NumPy 项目维护。
NOTICE 记录论文、Strassen 及 NumPy 的来源。
测试由 AI 辅助执行，测试通过不意味着人类同行评审或论文完整定理的独立验证。

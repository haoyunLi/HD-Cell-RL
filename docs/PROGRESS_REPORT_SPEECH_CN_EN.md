# From 2 µm bins to physical cells

## Bilingual presentation script / 中英文双语讲稿

数据更新：2026-09-08。最新方法结果测于 09-03；最新真实性审计完成于 09-07 本地时间 / 09-08 UTC。正式 PPO 仍为保留的 8 月 overfit run。

分步演示更新：05、17、19、24、27 页支持 Next / Back / Replay；10 页在同一图像上揭示扩张轮廓。按方向键也可推进。所有演示默认不自动播放。22、25 页的完整数值表可展开；现场不必遍历四个 patch，可留作问答。

对应报告：<https://haoyunli.github.io/HD-Cell-RL/progress-report/>

建议总时长约 55 分钟，另留 5 分钟问答。每页的中文和英文内容等价，现场选择一种语言讲即可，不需要把两种语言连续念一遍。方括号中的内容是动作提示，不需要读出来。

The suggested running time is about 55 minutes, followed by 5 minutes for questions. The Chinese and English scripts are equivalent alternatives. Choose one language for the presentation rather than reading both. Text in square brackets is a stage direction and should not be spoken.

## Timing / 时间安排

| Section | Slides | Time |
|---|---:|---:|
| Motivation and biological problem / 背景与生物学问题 | 00–04 | 9.5 min |
| Simulation and ground truth / 模拟数据与真值 | 05–15 | 18.5 min |
| EM and RL architecture / EM 与 RL 架构 | 16–19 | 10 min |
| Historical training, latest comparison and realism audit / 历史训练、最新比较与真实性审计 | 20–28 | 17 min |

---

## Slide 00 — From 2 µm bins to physical cells

**Suggested time / 建议时间：1 minute**

### 中文

大家好，今天我想介绍这个项目目前的进展。它要解决的问题可以用一句话概括：Visium HD 测到的是大量 2 微米的方形 barcode，但我们真正关心的是组织里的物理细胞。那么，怎样判断每一个 2 微米 bin 应该属于哪一个细胞，并最终重建出每个细胞的表达谱？

这个问题看起来像 segmentation，但它不只是画边界。图像告诉我们形态，RNA 告诉我们分子组成，两种证据在细胞边界附近经常不一致。因此，我把任务定义为 molecular ownership，也就是推断每个物理 barcode 的细胞归属。今天我会先解释为什么需要做 cell-level reconstruction，然后介绍模拟数据、EM soft assignment、patch-level RL，以及目前的训练和评估结果。

### English

Hello everyone. Today I will present the current progress of this project. The central question is simple to state: Visium HD measures a very large number of square, two-micrometre barcodes, but the biological objects we care about are physical cells. How can we decide which cell owns each two-micrometre bin and reconstruct one expression profile for each cell?

This may sound like a segmentation problem, but it is more than drawing a boundary. The image gives us morphology, while RNA gives us molecular identity, and these two sources of evidence often disagree near cell boundaries. I therefore frame the task as molecular ownership: inferring the physical cell that owns each barcode. I will first explain why cell-level reconstruction is needed, then walk through the simulation, the generalized EM soft assignment, the patch-level reinforcement-learning refinement, and our current training and evaluation results.

---

## Slide 01 — What is Visium HD?

**Suggested time / 建议时间：2.5 minutes**

### 中文

[依次指向 H&E、2 µm 网格和 gene-count vector。]

Visium HD 把三类信息放在同一个坐标系统里。第一类是 H&E 图像，我们可以看到组织形态和细胞核。第二类是连续铺设的 2 × 2 微米 barcode 网格。第三类是每个 barcode 上测到的基因计数向量。所以它的优势不是只有空间分辨率高，而是形态、位置和表达是对齐的。

和普通 Visium 的 55 微米 spot 相比，2 微米网格更有机会保留拥挤组织中的细胞边界。一个 55 微米 spot 往往覆盖多个细胞，而一个 2 微米 bin 通常只覆盖细胞的一小部分。和 Xenium Prime 这类 targeted panel 相比，这个数据保留了大约 18,132 个对齐的 panel features，而不是预先限制在约 5,000 个靶基因。和 scRNA-seq 相比，它没有在解离过程中丢失组织位置，而且能直接与 H&E morphology 对齐。

但这里有一个重要的 tradeoff：2 微米是很好的 measurement unit，却不是完整的 biological unit。每个小格子只含有很少的 RNA，一个细胞会被切成许多格子，而边界格子还可能混合两个细胞。所以下一步不是简单地分析所有 2 微米 bin，而是把它们重新组合成物理细胞。

### English

[Point to the H&E image, the two-micrometre grid, and the gene-count vector in sequence.]

Visium HD places three kinds of information in one coordinate system. First, we have the H&E image, which shows tissue morphology and nuclei. Second, we have a continuous grid of two-by-two-micrometre barcoded squares. Third, every barcode has a gene-count vector. Its advantage is therefore not only high spatial resolution. Morphology, location, and expression are aligned in the same field.

Compared with the 55-micrometre spots in conventional Visium, the two-micrometre grid has a much better chance of preserving boundaries in crowded tissue. A conventional spot often covers several cells, whereas a two-micrometre bin usually captures only a small part of a cell. Compared with a targeted platform such as Xenium Prime, this dataset retains about 18,132 aligned panel features rather than restricting the experiment to a preselected panel of roughly 5,000 genes. Compared with dissociated single-cell RNA sequencing, the spatial position is retained and can be aligned directly with H&E morphology.

The tradeoff is that two micrometres is an excellent measurement unit, but it is not a complete biological unit. Each square contains little RNA, one cell is split across many squares, and a boundary square may contain material from two cells. The next task is therefore to reconstruct physical cells rather than treating every two-micrometre bin as an independent cell-like object.

---

## Slide 02 — Why not stop at 2, 8, or 16 µm bins?

**Suggested time / 建议时间：2.5 minutes**

### 中文

这里比较三种固定大小的方格。原始 2 微米 bin 保留了最高的空间细节，但它的表达分散在上万个基因中，单个基因的证据仍然稀疏。最新全片统计中，组织内 96.20% 的 bins 有计数，中位数是 21 UMI，所以不能把稀疏简单解释成“大多数 bin 全零”。同一个细胞的表达被拆散在很多小格子里，直接做 cell typing 会非常不稳定。

把 2 微米 bin 合并成 8 微米以后，平均转录本数量大约提高 16 倍，信号更稳定。但是固定的正方形不会沿着真实细胞边界弯曲，所以边缘的 8 微米 bin 很容易跨过两个细胞。再合并到 16 微米，计数继续增加，但一个方格同时覆盖多个细胞的概率也更高。换句话说，更大的 bin 解决了 sparsity，却用更严重的 cell mixing 作为代价。

我们真正需要的不是在 2、8 和 16 微米之间选一个最好的正方形，而是改变 aggregation 的形状。右边的目标是 nucleus-anchored、大小可变、可以沿细胞边界生长的 cell-shaped region。这样聚合以后，一个统计单位才更接近一个物理细胞，而不是某个任意大小的图像方格。

### English

This slide compares three fixed square sizes. Native two-micrometre bins preserve fine spatial detail, but their counts are spread over thousands of genes, leaving sparse evidence per gene. In the latest whole-slide audit, 96.20% of tissue bins are nonzero and the median is 21 UMIs. Sparse gene-level evidence does not mean that most tissue bins are entirely empty. The expression of one cell is fragmented across many small squares, so treating each bin as a cell-like unit produces unstable cell typing.

Aggregating to eight micrometres gives roughly sixteen times more transcripts on average and improves the signal. However, a fixed square does not bend around a biological boundary, so an eight-micrometre bin near an edge can easily cross two cells. At sixteen micrometres, the count density increases again, but the probability of combining several physical cells is even higher. Larger bins reduce sparsity by accepting more cell mixing.

The real goal is therefore not to select the best square size among two, eight, and sixteen micrometres. We need to change the shape of the aggregation unit. The target on the right is a nucleus-anchored, variable-sized, cell-shaped region that can follow a physical boundary. After reconstruction, one statistical unit should correspond more closely to one physical cell rather than to an arbitrary square in the image.

---

## Slide 03 — Reassemble measurements into physical-cell profiles

**Suggested time / 建议时间：1.5 minutes**

### 中文

左边是现在的数据形式：很多稀疏的 bin-level count vectors，其中一部分位于清楚的细胞内部，另一部分位于边界，归属不确定。中间是我们希望得到的结果：每个 barcode 有一个最终的物理 owner，每个 cell 汇总出一个表达向量。

这样做会改变后续分析的统计单位。我们可以在 cell level 做 cell annotation、differential expression、cell–cell communication，也可以把 morphology 和 expression 放到同一个细胞对象上分析。需要强调的是，最终输出要求 unique owner，但推断过程中不应该过早丢掉 uncertainty。后面介绍的 EM 会先保留 soft probability，RL 再对高不确定性的 bin 做离散修正。

### English

The left side shows the current measurement space: many sparse bin-level count vectors. Some bins sit clearly inside a cell, while others lie near boundaries and have uncertain ownership. The middle shows the target representation: every physical barcode has one final owner, and each reconstructed cell has one aggregated expression vector.

This changes the statistical unit used downstream. We can perform cell annotation, differential expression, and cell–cell communication at the cell level, and we can associate morphology and expression with the same physical object. The final result requires a unique owner for every barcode, but we should not discard uncertainty too early. The EM stage will retain a soft ownership distribution, and the RL stage will later make discrete corrections for bins with high uncertainty.

---

## Slide 04 — The molecular-ownership challenge

**Suggested time / 建议时间：2 minutes**

### 中文

一条看起来合理的图像边界，并不一定能正确回答 RNA 来自哪个细胞。边界附近的 RNA 可能发生扩散或 spillover；组织原本是三维结构，切片后多个细胞可能投影到同一个二维区域；细胞形状本身也可能非常复杂。再加上低计数，单个 bin 的表达证据会很弱。

如果边界 RNA 分错，影响会向下游传播。首先得到 mixed cell profiles，然后可能产生错误的 cell type，再进一步影响 neighborhood 和 cell–cell communication。因此，这个项目不把 image segmentation 当作完整答案。Nuclear segmentation 提供稳定的物理 anchor，但非核区域的 RNA ownership 仍然需要结合 expression 和 geometry 来推断。这也是后面需要 soft assignment 和 patch-level optimization 的原因。

### English

A visually plausible boundary does not necessarily identify the cell that generated the RNA near that boundary. RNA can be displaced by diffusion or spillover. The tissue is three-dimensional, while the section is two-dimensional, so several cells may project into the same local area. Cell shapes can also be irregular, and low-count bins provide weak expression evidence.

An ownership error then propagates downstream. It first produces mixed cell profiles, which can bias cell-type annotation and eventually distort neighborhood or cell–cell communication analysis. For this reason, the project does not treat image segmentation as the complete answer. Nuclear segmentation gives us stable physical anchors, but RNA ownership outside the nucleus must still be inferred from expression and geometry. This motivates the soft-assignment stage and the later patch-level optimization.

---

## Slide 05 — Build an answer key, then hide it from inference

**Suggested time / 建议时间：2 minutes**

### 中文

[按 Next 依次展示核、模拟 mask、RNA 放置、bin 聚合和隐藏真值。]

先问为什么需要 simulation：真实 HD 给了图像、核和 bin counts，但没有每个 bin 的完整来源答案。如果不知道答案，就很难判断一个看起来合理的分配是否真的正确。因此我生成一套可追溯的数据，并保留来源标签作为评价用的答案。

这里 A–E 是固定位置的示意细胞，后面的 EM 页会重复使用它们。先建立 nuclear anchors，再生成 cell masks；这些 whole-cell masks 是模拟定义，不是已经验证的真实细胞边界。随后为每个物理细胞生成表达，按不同 compartment 放置 RNA，再聚合成 2 µm bins。图中的点只解释空间放置，不代表精确的 UMI 数量。

最后把来源标签藏起来。算法只能看到 nuclei、bin counts 和独立的 scRNA reference；只有评价阶段才能打开 GT。最新数据中 generator 和 reference 已按 donor 分开。下一组 slides 再逐步解释真实项目如何实现这些步骤。

### English

[Use Next to reveal nuclei, simulated masks, RNA placement, bin aggregation, then the hidden answer key.]

Why simulate? Real HD gives us an image, nuclei and bin counts, but not a complete answer about which cell generated each bin's expression. Without that answer, a plausible assignment is difficult to verify. The simulator therefore retains source labels as an evaluation answer key.

A–E are fixed illustrative cells that will reappear in the EM demonstration. Nuclear anchors come first, followed by simulator-defined whole-cell masks, cell-specific expression, compartment-biased RNA placement and aggregation into two-micrometre bins. The whole-cell masks are not validated real boundaries, and the dots illustrate placement rather than exact UMI counts.

Finally, hide the source labels. Inference sees nuclei, bin counts and an independent scRNA reference; only evaluation opens the GT answer key. Generator and reference donors are separated in the latest dataset. The following slides explain how the actual project implements these stages.

---

## Slide 06 — Start from physical nuclei

**Suggested time / 建议时间：2 minutes**

### 中文

第一步是在 colorectal H&E 上得到可靠的 nuclear instances。我使用 StarDist 的 `2D_versatile_he` 模型，把整张图切成有 overlap 和 context 的 tiles，减少 tile boundary artifacts。预测时 probability threshold 是 0.60，NMS threshold 是 0.30。

模型最初检测到 170,252 个 nuclei。之后按组织区域和面积做 QC，保留面积在 5 到 400 平方微米之间的对象，最后得到 167,509 个 retained nuclei。这里可以回答一个常见问题：为什么不能简单 filter empty nuclei？如果一个 nucleus 当前没有被少量 2 微米 reads 覆盖，它仍然可能是图像中真实存在的物理细胞，也可能在邻近非核区域有 RNA。直接按当前 count 是否为空过滤，会把 capture sparsity 当成“没有细胞”。所以应该按 segmentation quality 和形态做 QC，而不是仅按表达是否为空删除。

这些 retained nuclei 在后面的模拟、EM 和 RL 里都作为不可跨越的 physical seeds。

### English

The first step is to obtain reliable nuclear instances from the colorectal H&E image. I use the StarDist `2D_versatile_he` model and process the whole image in overlapping tiles with context to reduce boundary artifacts. The probability threshold is 0.60 and the non-maximum-suppression threshold is 0.30.

The model initially detects 170,252 nuclei. After tissue and area quality control, retaining objects between 5 and 400 square micrometres, 167,509 nuclei remain. This also explains why I do not simply filter every nucleus with no observed RNA. A nucleus with no count in the small set of overlapping two-micrometre bins may still be a real physical cell in the image, and that cell may have RNA in nearby non-nuclear regions. Removing it based only on an empty count would confuse capture sparsity with absence of a cell. Nuclear objects should be filtered by segmentation quality and morphology, not only by whether expression happens to be observed.

These retained nuclei become protected physical seeds in the later simulation, EM, and RL stages.

---

## Slide 07 — Assign a cell type to each nucleus

**Suggested time / 建议时间：2.5 minutes**

### 中文

下一步不是随机给 nucleus 分 cell type，而是尽量保留真实组织里的 spatial pattern。我先用 colorectal scRNA reference 对真实 8 微米空间 bin 做 STHD deconvolution。每个 8 微米 bin 得到的是一个对所有 eligible cell types 的完整 posterior vector，不是一个单独的 label。

[先指向左侧网格中的多个 nuclei，再指向橙色格子里的 selected nucleus。]

每个 nucleus 内的 2 微米中心点找到它所属的 8 微米 parent bin，并继承这个 bin 的完整 posterior。然后对同一个 nucleus 的所有向量求平均，最后才取平均 posterior 的 argmax 作为 cell type。这里没有用 0.48 或其他单一数值做 threshold。

图里的例子来自当前数据中的 nucleus 80,653。12 个 nuclear sample centers 平均以后，TA 是 0.695，Goblet 是 0.285，Immature Goblet 是 0.020，所以最终选择 TA。全数据得到 36 个 cell types，只有 3 个 nucleus 需要 nearest-bin fallback。这样生成的 type map 仍然来自真实组织表达，而不是人为随机铺设。

### English

The next step is not to assign cell types randomly. I want to preserve the spatial pattern observed in real tissue. I first use the colorectal single-cell reference to deconvolve the real eight-micrometre spatial bins with STHD. Each eight-micrometre bin receives a complete posterior vector over all eligible cell types, not a single hard label.

[Point first to the different nuclei on the grid, then to the selected nucleus inside the orange bin.]

Every two-micrometre center inside a nucleus finds its parent eight-micrometre bin and inherits that full posterior vector. I average all posterior vectors observed within the same nucleus and only then take the argmax as the assigned cell type. No single number such as 0.48 is used as an independent threshold.

The example on the slide is nucleus 80,653 from the current dataset. After averaging twelve nuclear sample centers, the posterior is 0.695 for transit-amplifying cells, 0.285 for Goblet cells, and 0.020 for Immature Goblet cells, so the final label is TA. Across the dataset, this produces 36 cell types, with only three nearest-bin fallbacks. The simulated type map therefore retains real spatial expression structure rather than placing cell types randomly.

---

## Slide 08 — Cell type sets the target cell area

**Suggested time / 建议时间：1.5 minutes**

### 中文

只有 nucleus 还不能确定 whole-cell boundary，所以我用 measured nuclear area 和 type-specific nucleus-to-cell area ratio 估计 target cell area。不同 cell type 的 nucleus 占整个细胞的比例不同。例如 epithelial cell 的 nucleus 大约占 14% 到 30%，fibroblast 大约占 8% 到 16%，而 T 或 B cell 可以达到 50% 到 75%。

因此，即使一个 fibroblast 和一个 lymphocyte 的 nucleus 面积相近，也不应该把它们扩成同样大的 whole-cell mask。这里的 ratio 是 simulation benchmark 使用的可重复 prior，不应该被理解成适用于所有组织和实验条件的生物学常数。它的作用是提供一个合理的 capacity target，避免所有 cell 都以同样半径生长。

### English

A nucleus alone does not determine the whole-cell boundary, so I estimate a target cell area from the measured nuclear area and a cell-type-specific nucleus-to-cell area ratio. These ratios differ by cell type. In this benchmark, an epithelial nucleus occupies roughly 14 to 30 percent of the cell area, a fibroblast nucleus about 8 to 16 percent, and a T or B cell nucleus about 50 to 75 percent.

Therefore, a fibroblast and a lymphocyte with similarly sized nuclei should not be expanded to the same whole-cell area. These ratios are reproducible priors used for the simulation benchmark. They are not universal biological constants. Their role is to provide a plausible capacity target so that every cell is not grown with the same fixed radius.

---

## Slide 09 — One nucleus anchors one cell region

**Suggested time / 建议时间：2.5 minutes**

### 中文

Whole-cell growth 从 retained nucleus 开始。每个 retained nucleus 初始化一个独立的 cell ID，所有 nuclear pixels 永远保留这个 ID。也就是说，一个 growing cell 不能穿过或吸收另一个 retained nucleus，因此不会出现两个 retained nuclei 被合并到同一个模拟 cell 的情况。

对于非核 image pixels，算法按到 nuclei 的精确距离排序，让候选 cell fronts 竞争。每个 cell 达到 type-specific target area 或 25 微米最大距离后停止。如果周围过于拥挤，cell 会被记录为 blocked，而不是穿过邻居强行达到面积。最终每个 H&E image pixel 只有一个 cell label。

这里要区分 image-pixel ownership 和 barcode ownership。图像 pixel 是互斥的，不代表一个 2 微米 barcode 一定只覆盖一个 cell。Barcode 自身有面积，它可以跨过两个相邻 cell regions。当前 multi-owner mapping 中有 694,393 个 assigned bins，也就是 13.95%，具有多个 fractional ground-truth owners。这正是后面评估需要 fractional GT 的原因。

### English

Whole-cell growth begins from the retained nuclei. Every retained nucleus initializes one independent cell ID, and every nuclear pixel permanently keeps that ID. A growing cell therefore cannot cross or absorb another retained nucleus. In this simulation, two retained nuclei cannot be merged into one simulated cell.

For non-nuclear image pixels, candidate cell fronts compete according to their exact distances to the nuclei. A cell stops growing when it reaches its type-specific target area or the maximum distance of 25 micrometres. If the tissue is too crowded, the cell is recorded as blocked rather than being forced through its neighbors. At the end of this step, every H&E image pixel has exactly one cell label.

It is important to distinguish image-pixel ownership from barcode ownership. Exclusive image pixels do not imply that every two-micrometre barcode covers only one cell. A barcode has physical area and can cross two adjacent cell regions. In the current multi-owner mapping, 694,393 assigned bins, or 13.95 percent, have more than one fractional ground-truth owner. This is why the evaluation later needs fractional ground truth.

---

## Slide 10 — Actual expansion on the same image

**Suggested time / 建议时间：1.5 minutes**

### 中文

[先选定一个细胞，按一次向右键显示 expanded masks，再按向左键返回。]

这是项目保存的真实输出，不是 AI 生成的 expansion。底图、区域和比例尺不变：先看洋红色的 nuclear boundaries，再在同一位置显示青色的 expanded cell boundaries。这样不需要在两张图之间重新寻找同一个细胞。

但“真实输出”不等于“真实 whole-cell GT”。这仍是从 nuclear seeds 生长出来的模拟 cell masks。原始全片中 115,179 个细胞达到目标面积，52,330 个受到拥挤阻挡。重要的是阻挡会被记录，而不是穿过相邻细胞继续增长。

### English

[Choose one cell, advance once to reveal expanded masks, then go back to the nuclear view.]

These are saved project outputs, not AI-generated expansion. The field and scale stay fixed: magenta nuclear boundaries are followed by cyan expanded boundaries in exactly the same image. You do not have to locate the same cell again in a second panel.

Actual output does not mean validated real whole-cell ground truth. These are still simulator-defined masks grown from nuclear seeds. In the original whole slide, 115,179 cells reached their target area and 52,330 were blocked by crowding. The simulator records that limitation rather than expanding through neighboring cells.

---

## Slide 11 — Fractional ownership inside a barcode

**Suggested time / 建议时间：2 minutes**

### 中文

为了计算 barcode-level ground truth，我不能只看 bin center 落在哪个 cell 里。一个 2 × 2 微米方格有真实面积，所以我在每个 barcode 内做 sub-bin sampling。图中这个例子有 9 个采样点，其中 6 个位于 Cell A，3 个位于 Cell B，因此 ground truth 是 A 等于 6/9、B 等于 3/9。

这个表示同时保存三个信息。Coverage fraction 表示 barcode 面积中有多少属于每个 cell；compartment fraction 记录覆盖的是 nucleus、cytoplasm 还是 membrane；physical identity 确保 barcode 只存一次。这里不会为了每个候选 cell 复制一条 observation。于是，同一个 physical barcode 可以有多个 fractional GT owners，但在数据结构里仍然是唯一对象。这修复了早期 pseudo data 中“所有 bin 都只有一个 ground-truth owner”的问题。

### English

To compute barcode-level ground truth, I cannot simply ask which cell contains the bin center. A two-by-two-micrometre square has physical area, so I perform sub-bin sampling inside each barcode. In the example, six of nine sample points fall inside Cell A and three fall inside Cell B. The fractional ground truth is therefore six ninths for A and three ninths for B.

This representation preserves three types of information. The coverage fraction records how much of the barcode area belongs to each cell. The compartment fraction records whether that area is nuclear, cytoplasmic, or membrane. Physical identity ensures that the barcode is stored only once rather than duplicated for every candidate cell. One physical barcode can therefore have several fractional ground-truth owners while remaining a unique observation in the data structure. This corrects the earlier pseudo-data problem in which every bin effectively had only one ground-truth owner.

---

## Slide 12 — Separate expression generation from inference

**Suggested time / 建议时间：2 minutes**

### 中文

这一页改用最新 donor-disjoint 局部数据的数字，不再把原始全片和新实验混在一起。Generator 使用 5 个 donors 的 4,500 个 scRNA cells，scDesign3 为 233 个实际贡献表达的 physical cells 生成 profiles。推断用的 reference 则来自另外 7 个 donors、3,596 个 cells；donor 和 scRNA barcode overlap 都为零。

这里 233 个 contributing cells、190 个 context nuclei 和 43 个 scored core cells 是三种范围。边缘以外的细胞也可能给局部 bins 贡献表达，但并不都属于 core evaluation。Reference 保留 10 个类型、15,143 个选定基因；HD 输出面板是 18,132 个 features，不要把两个维度混用。

Donor 切分已经完成，但仍在同一 cohort 内，空间形态也来自同一张 colorectal slide。它不是独立组织验证。当前 scDesign3 拟合主要按 cell type 建模，后面会看到它的 library-size variation 仍有问题。

### English

This slide now uses the latest donor-disjoint local dataset rather than combining it with the original whole-slide totals. Five generator donors contribute 4,500 scRNA cells, and scDesign3 produces profiles for 233 physical cells that contribute expression. The inference reference uses 3,596 cells from seven different donors. Donor overlap and scRNA barcode overlap are both zero.

The 233 contributing cells, 190 context nuclei and 43 scored core cells are different populations. Cells outside the context can still contribute expression near its edge without becoming core evaluation targets. The inference reference retains ten types and 15,143 selected genes; the HD output panel has 18,132 features.

Donor separation is complete, but this remains within one cohort and uses geometry from the same colorectal tissue slide. It is not external tissue validation. The current scDesign3 fit is largely type-conditioned, and later slides examine its remaining library-size distortion.

---

## Slide 13 — Place molecules into biological compartments

**Suggested time / 建议时间：2.5 minutes**

### 中文

一个 cell-level count vector 还没有空间位置，所以我把每个 gene 的 molecules 分到 nucleus、cytoplasm 和 membrane 三个 compartments。

第一步是 nuclear draw。每个 gene 根据 intron/exon evidence 得到 nuclear probability，再通过 binomial draw 决定多少 molecules 留在 nucleus。第二步对剩余 molecules 做 boundary routing，membrane 和 secretory genes 会获得更高的 boundary probability。第三步，未进入 nucleus 或 membrane 的 molecules 留在 cytoplasm。这个过程还保留同一 cell type 内不同 physical cells 之间的变化。

最后三个 compartment 使用不同的 spatial kernels。Nuclear molecules 更偏向 nuclear center；cytoplasmic molecules 使用普通的 cell-interior distribution；membrane kernel 反转 radial preference，让 molecules 更靠近 cell boundary。右边的 25.0%、40.2% 和 34.8% 是当前模拟完成后观察到的整体输出比例，不是预先硬编码成这三个固定百分比。真正控制分配的是前面的 gene-specific probabilities 和随机 draws。

### English

A cell-level count vector still has no spatial coordinates, so I divide each gene's molecules among nuclear, cytoplasmic, and membrane compartments.

The first stage is the nuclear draw. Intron and exon evidence defines a gene-specific nuclear probability, and a binomial draw determines how many molecules remain in the nucleus. The second stage routes molecules near the boundary. Membrane and secretory genes receive additional probability in the boundary band. Molecules that are assigned to neither the nucleus nor the membrane remain in the cytoplasm. The procedure also preserves variation among physical cells of the same type.

Finally, each compartment uses a different spatial kernel. Nuclear molecules are biased toward the nuclear center. Cytoplasmic molecules use the normal interior distribution. The membrane kernel reverses the radial preference so that molecules accumulate closer to the cell boundary. The 25.0, 40.2, and 34.8 percent values on the right are aggregate output proportions observed after the current simulation. They are not three hard-coded allocation percentages. The actual assignments come from gene-specific probabilities and stochastic draws.

---

## Slide 14 — Return to native Visium HD barcodes

**Suggested time / 建议时间：2 minutes**

### 中文

分配完细胞内分子位置后，再按原始 Visium HD 网格聚合。最新局部数据有 10,827 个唯一 physical barcodes、18,132 个 HD features 和 233 个 contributing cells。旧的 971.2 million counts 是原始全片的量，不能用来描述这个局部实验。

纯 synthetic cell RNA 的来源可用于 GT，但方法推断只看聚合矩阵和核信息。后面的 realism controls 还会加入真实组织表达，这部分没有完整的 cell-owner GT，必须单独标成 unknown，不能为了方便评估硬塞给已知细胞。Count conservation 也需要区分裁剪前完整细胞计数和裁剪后的局部输出。

### English

After placing molecules within cells, we aggregate them onto the native Visium HD grid. The latest local dataset has 10,827 unique physical barcodes, 18,132 HD features and 233 contributing cells. The old 971.2 million count total belongs to the original whole-slide simulation, not this local experiment.

Synthetic cell RNA retains source identities for evaluation, while inference receives the aggregated matrix and nuclear observations. Some later realism controls add real tissue expression, whose cell ownership is unknown. That component must remain separately labelled rather than receiving invented cell-owner GT. Count conservation also needs to distinguish complete cells before cropping from the local output.

---

## Slide 15 — Fractional truth, discrete final state

**Suggested time / 建议时间：1.5 minutes**

### 中文

评估和 inference state 使用两种不同但兼容的表示。Ground truth 是 fractional 的，因为一个 physical barcode 可以覆盖多个真实 cell masks，例如 A 是 0.58、B 是 0.42。这个比例用于计算 fractional IoU 和 boundary accuracy。

但最终 HD-Cell-RL state 是 discrete 的。每个 barcode 只有一个 `cell_id`，而且在整个 patch 内保持 globally unique。这和现有 REPLACE environment 一致，也避免把一个 barcode 的完整 count vector 重复加到多个 cells。当前数据有大约 4.98 million assigned bins，其中 694 thousand 是 multi-owner bins，占 13.95%，单个 barcode 最多覆盖 5 个 owners。所有七种方法的最终输出都会经过 unique-owner audit。

### English

Evaluation and the inference state use two different but compatible representations. Ground truth is fractional because one physical barcode can cover several true cell masks, for example 0.58 for Cell A and 0.42 for Cell B. These fractions are used for metrics such as fractional IoU and boundary accuracy.

The final HD-Cell-RL state remains discrete. Every barcode has one `cell_id`, and ownership is globally unique across the patch. This matches the existing REPLACE environment and avoids adding the complete count vector of one barcode to several cells. The current dataset contains about 4.98 million assigned bins. Roughly 694 thousand are multi-owner bins, representing 13.95 percent of assigned bins, and a barcode can overlap at most five owners. The final outputs of all seven methods are checked with the same unique-owner audit.

---

## Slide 16 — HD-Cell-RL architecture

**Suggested time / 建议时间：2 minutes**

### 中文

现在进入方法本身。整体架构是先做 probabilistic assignment，再做 discrete refinement。

第一步是 confident nuclear seeds，它们是 locked physical anchors。第二步是 patch-level generalized EM，为每个 unique physical barcode 计算对候选 cells 的 soft probability。第三步用 normalized entropy 衡量完整 ownership distribution 的不确定性。低 entropy bin 接受 EM top-1，高 entropy bin 才进入 RL 的候选 action mask。第四步，patch RL 用离散 REPLACE 把一个 barcode 从当前 owner 移到另一个候选 cell。最后输出仍然是每个 barcode 一个 unique owner。

候选距离直接复用现有 episode construction 的 20 微米 MaxDis。每个 bin 考虑 20 微米内的所有 nuclei，没有 nearest-three restriction。Core cells 和 margin cells 都能竞争，避免 patch edge 人为丢失 owner。Expression likelihood 使用 scRNA reference。EM probability 不直接加入 RL reward，它只用于 initialization、entropy 和 action masking，避免 expression 与 distance 被重复计算。

### English

We now move to the assignment method itself. The architecture performs probabilistic assignment first and discrete refinement second.

The first stage is a set of confident nuclear seeds, which remain locked physical anchors. The second stage is patch-level generalized EM, which computes a soft ownership distribution over candidate cells for each unique physical barcode. The third stage uses normalized entropy to measure uncertainty in that complete distribution. Low-entropy bins keep the EM top-one owner, while high-entropy bins enter the RL action mask. In the fourth stage, patch-level RL uses a discrete REPLACE action to move one barcode from its current owner to another candidate cell. The final output still has one unique owner per barcode.

Candidate construction directly reuses the existing 20-micrometre MaxDis from episode building. Every bin considers all nuclei within 20 micrometres, with no nearest-three restriction. Core and margin cells both participate, preventing patch boundaries from removing legitimate owners. Expression likelihood uses the single-cell RNA reference. EM probability is not added directly to the RL reward. It affects initialization, entropy, and action masking, which avoids double-counting the same expression and distance evidence.

---

## Slide 17 — Why the nearest nucleus is not always the owner

**Suggested time / 建议时间：3 minutes**

### 中文

[四步演示：Distance → Expression → All bins together → Update cell types。]

现在回到 A–E。它们是五个物理细胞，不是五种 cell types。选中的 bin b 离 A 更近，所以只看距离时，A 的概率稍高。MaxDis 内所有合法候选都参与，没有 nearest-3。

[第二步。] 这个 bin 的表达更符合 B。低 expression confidence 时变化较弱，高 confidence 时 B 的概率明显增加。变化的是我们的 ownership 判断，不是 RNA 在细胞间移动。这是可重复计算的教学例子，不是一个真实 EM run；示意中的等权 type/cell blend 也不是当前实验拟合出的权重。

[第三步。] 固定同一组 q，把所有 bins 的 responsibilities 一次算完，并进行 damping。左边每个小格的彩条是一个 soft ownership distribution，不是多个重复 observations。Nuclear rows 始终 exact one-hot。Normalized entropy 只衡量不确定性；后续 top-1 hardening 建立单独的 discrete RL state。

[第四步。] 完整 R 出来之后，M-step 才更新所有 cell-type posteriors，旧证据不会跨迭代反复累加。这里必须区分 q 的 cell-type uncertainty 与 r 的 physical-cell ownership uncertainty。当前 baseline 固定的是 nuclear cell-specific profile，不是 q 或 r；它们仍交替更新。真实比较用 α=1、β=1.5、κ=2、profile_update_damping=0，background off，暂未重新校准 entropy 或训练 PPO。

### English

[Step through Distance → Expression → All bins together → Update cell types.]

Return to A–E: five physical cells, not five cell types. Bin b is closer to A, so distance alone slightly favors A. Every valid candidate inside MaxDis competes; there is no nearest-three restriction.

[Step two.] The bin's expression is more compatible with B. Low expression confidence changes the probabilities weakly; high confidence gives B much more support. Our ownership belief changes, not the physical location of RNA. This is a deterministic teaching example, not a recorded EM run. Its equal type/cell blend is illustrative, not a fitted experiment parameter.

[Step three.] Hold q fixed, compute all bin responsibilities, then apply damping. Colored strips within a square form one soft ownership distribution, not duplicated observations. Nuclear rows stay exactly one-hot. Normalized entropy measures uncertainty; later top-one hardening initializes a separate discrete RL state.

[Step four.] Only after the entire R is complete does the M-step update all cell-type posteriors, recomputing rather than repeatedly accumulating old evidence. Distinguish uncertainty about cell type, q, from uncertainty about physical ownership, r. The current baseline freezes nuclear cell-specific profiles, while q and r still alternate. The measured comparison uses alpha one, beta 1.5, kappa two, profile update damping zero and no background component. Entropy recalibration and PPO remain paused.

---

## Slide 18 — What each reward term asks

**Suggested time / 建议时间：3.5 minutes**

### 中文

在讲 policy loop 之前，先解释 global objective 的组成。每个 reward term 回答一个不同的生物学问题。

`w1` 是 new expression confidence：把 bin 加给 cell 以后，重新计算的 cell-type posterior 是否更明确？`w5` 是 existing-posterior compatibility：不要求 posterior 变尖，而是问这个 bin 的 expression 是否和 cell 当前的 type posterior 一致。这两个都是 expression terms，但含义不同。`w1` 更关注 posterior 改变，`w5` 更关注 candidate 与当前 cell identity 的相容性。

`w2` 是 nucleus distance penalty，owner nucleus 越远，惩罚越大。`w3` 是 competing-nucleus 或 overlap penalty，如果另一个 nucleus 比当前 owner nucleus 更靠近 candidate bin，就惩罚当前 assignment。`w4` 是 eight-neighbor support，鼓励同一个 cell 的 bins 在局部保持连接。Shape prior 比较 reconstructed cell 的 log area、compactness、solidity 和 anisotropy 是否像 reference cells。

一次 REPLACE 会同时影响 source cell、target cell 和整个 patch，所以 reward 不是单个局部 feature，而是 `J_after - J_before`。正项提供 evidence，penalty 项扣分。这里展示的是所有可用 components；真正的 preserved checkpoint 哪些项开启、权重是多少，在下一页给出。这个区分很重要，因为概念上最重要的 term 不一定在当前 overfit checkpoint 中被启用。

### English

Before describing the policy loop, I want to explain the components of the global objective. Each reward term asks a different biological question.

`w1` measures new expression confidence: after assigning the bin to a cell and recomputing the cell-type posterior, does the cell identity become clearer? `w5` measures compatibility with the existing posterior: without requiring the posterior to become sharper, does the bin's expression agree with the cell's current type distribution? Both are expression terms, but they have different meanings. `w1` focuses on how the posterior changes, while `w5` focuses on compatibility with the current cell identity.

`w2` is the nucleus-distance penalty, which increases as the bin moves farther from its owner nucleus. `w3` is the competing-nucleus or overlap penalty. It penalizes the current assignment when another nucleus is closer to the candidate bin. `w4` is eight-neighbor support, encouraging locally connected cell regions. The shape prior evaluates whether log area, compactness, solidity, and anisotropy resemble the reference-cell distribution.

A REPLACE action changes the source cell, the target cell, and the full patch. The reward is therefore not one local feature. It is the global difference, `J_after - J_before`. Positive terms add evidence and penalty terms subtract it. This slide shows all available components. The next slide shows which components and exact weights were active in the preserved checkpoint. That distinction matters because a biologically important term may still be disabled in the current overfit experiment.

---

## Slide 19 — A REPLACE can help the objective and hurt the answer

**Suggested time / 建议时间：3.5 minutes**

### 中文

[先不打开 GT。依次展示 current state、policy choice、REPLACE、global delta，最后 Reveal GT。]

左边的转移动画是示意，不是真实细胞形状；cell IDs、barcode、action rank 和 reward 分量来自 8 月保存的 rollout。这里 A/B 是该次动作的 source/target 角色，不是前一页示意细胞的真实身份。案例按原 CSV 顺序选择每类第一个 inside-core 且双方 GT 映射完整的动作，不是挑 reward 最大的例子。

State 包含当前 unique ownership、expression、type posterior、geometry、neighborhood 和 shape。Policy 在 legal actions 中选择一个离散 REPLACE，也可以 STOP。设计上的默认 mask 保护 nuclear bins、只开放高 entropy 非核 bins；历史输出本身不能被当成 9 月修复后的重新评估。右边的 immediate-reward rank 说明 policy 不一定选当前 delta 最大的动作。

执行 REPLACE 时，一个 barcode 从 A 完整移到 B。Environment 重建双方状态后，计算 whole-patch objective 的差值。图中已经是加权后的各项贡献，单位是 10 的负六次方；rounding 和 numeric residual 也保留，所以能对回真实 reward。距离项本身是 penalty，但一次替换如果减小距离惩罚，它的 delta 可以为正。

[最后揭示 GT，切换两个案例。] 一个动作纠正 ownership，另一个破坏原本正确的 ownership。两者都可能拿到正 reward。GT 不参与 inference 或 reward，只在这里检验结果。展开 checkpoint weights 可以看到旧 run 的精确配置；这正说明 objective 改善与生物学准确度不是同一件事。

### English

[Keep GT hidden. Advance through current state, policy choice, REPLACE and global delta; reveal GT last.]

The transfer drawing is schematic, not actual cell geometry. Cell IDs, barcode, action rank and reward components come from the saved August rollout. A/B now denote source and target roles, not the literal identities in the earlier teaching example. Each case is the first CSV row of its outcome that is inside-core and has both GT mappings, not the largest reward.

State includes unique ownership, expression, type posterior, geometry, neighbors and shape. The policy selects a legal discrete replacement or STOP. The default design protects nuclear bins and exposes high-entropy non-nuclear bins; these historical outputs are not a reevaluation after the September correctness fixes. The immediate-reward rank shows that the policy does not always choose the largest current delta.

REPLACE transfers one complete barcode from A to B. The environment rebuilds both cells and measures the whole-patch objective change. The plotted components are already weighted, in units of ten to the minus six. Rounding and numerical residuals are retained so their sum reconciles to the saved reward. Distance is a penalty, but reducing that penalty creates a positive delta.

[Reveal GT and switch cases.] One action corrects ownership; the other damages a previously correct assignment. Both can receive positive reward. GT enters neither inference nor reward: it is opened only for this evaluation. Expand the checkpoint-weight details for the exact historical settings. Improving the objective and improving biological accuracy are different outcomes.

---

## Slide 20 — Historical four-patch overfit training

**Suggested time / 建议时间：2 minutes**

### 中文

这里保留 8 月 25 日的正式 PPO overfit run，不把它伪装成最新数据上的新训练。它反复使用四个 development patches，每次 update 有 16 条 trajectories，总共 15 个 updates，历史 best 文件标成 update 10，使用 4 张 A40。

Patch 卡片展示 outer/core/context 的几何范围和各 patch 的 core/margin 数量。它证明训练管线可以运行，不证明跨 patch、donor 或真实组织的准确率。9 月 8 日已修复 checkpoint 评分时点与保存权重不一致的问题，但不会倒写旧 checkpoint；因此这里说“历史文件标签”，不声称它已经按修复后的逻辑重新挑选。

当前没有正式 PPO 新训练。后面先看冻结参数的 EM 方法比较，再看 pseudo 数据真实性审计。

### English

This is the preserved formal PPO overfit run from August 25, not a new training run on the latest data. It repeatedly used four development patches, with sixteen trajectories per update and fifteen updates in total. The historical best file is labelled update ten. The run used four A40 GPUs.

The cards distinguish outer, core and loaded-context geometry and list core/margin cell counts. This demonstrates a working training pipeline, not accuracy on unseen patches, donors or real tissue. A September 8 fix aligned checkpoint scoring with the saved policy state, but it does not rewrite historical checkpoints. The update-ten label is therefore historical, not a retrospectively corrected model selection.

No new formal PPO training has run. We next examine the frozen-parameter EM comparison and then the pseudo-data realism audit.

---

## Slide 21 — Five methods on new patches, with donors separated

**Suggested time / 建议时间：2 minutes**

### 中文

最新完成的方法比较是五种方法，不是旧七方法表。四个 test patches 在运行前按 metadata 选定，与 development 的 patch ID、context cell 和 rectangle 都不重叠，评估同一组 43 个 core cells。Generator/reference donors 已分开，但仍来自同一 cohort，几何仍是同一 slide。

Nucleus only 保留外部核种子；distance only 关闭 EM 表达项；frozen-profile EM 使用 scRNA 与固定核 profile；STCS 使用核表达与空间距离。Bin2Cell 需要特别说明：两个条件下 GEX StarDist 都检出零对象，所以这里实际是 external nuclear labels 加 max_bin_distance=5 的固定扩张。

SMURF 和 RL 没有在这次实验中重跑。这些区域在 9 月 3 日方法比较时是预先锁定的新区域，后来用于数据诊断，就不能再称为另一个全新 held-out test。

### English

The latest completed comparison contains five methods, not the old seven-method table. Four patches were selected from metadata before inference, with no overlap with development patch IDs, context cells or context rectangles. The same 43 core cells are evaluated. Generator and reference donors are separate, but remain within one cohort and use the same source-slide geometry.

Nucleus-only retains the supplied seeds; distance-only disables EM expression; frozen-profile EM uses scRNA and fixed nuclear profiles; STCS uses nuclear expression with spatial distance. Bin2Cell needs a configuration caveat: GEX StarDist detected zero objects in both conditions, so these results come from external nuclear labels plus fixed expansion at max_bin_distance five.

SMURF and RL were not rerun. The patches were unseen when locked for the September 3 method comparison. Their later reuse for data diagnostics is not another fresh held-out test.

---

## Slide 22 — Paired high-depth and stress-condition results

**Suggested time / 建议时间：2 minutes**

### 中文

[先指两组共用 0–1 刻度的点图，必要时展开完整数字。] 这次不能只给一个排名。高深度、没有添加背景时，EM mean core-cell IoU 是 0.911，Bin2Cell 0.878，distance 0.820，STCS 0.652，nucleus only 0.397。表达项在这个条件下有帮助。

匹配 capture 并加入旧代码称为 structured ambient 的未归属组织 RNA 后，Bin2Cell 是 0.675，STCS 0.642，distance 0.631，EM 0.629。EM 只是略低于 distance，差值约 0.002，不应夸大为显著差异。Gene Spearman 也明显下降。两个条件同时改变 depth 和 added RNA，因此不能仅凭此表确定其中哪一项导致下降。

这里展示标准 mean per-core-cell IoU；fractional IoU 在结果文件中单独保存。数字来自 9 月 3 日保存结果，未在 9 月 8 日工程修复后重跑，也不是后面 28 个 realism 配置上的工具表现。

### English

We should not collapse this experiment into one leaderboard. At high depth without added background, mean core-cell IoU is 0.911 for EM, 0.878 for Bin2Cell, 0.820 for distance-only, 0.652 for STCS and 0.397 for nucleus-only. Expression helps in this condition.

After capture matching and addition of unassigned tissue RNA, labelled structured ambient in the old pipeline, the scores are 0.675 for Bin2Cell, 0.642 for STCS, 0.631 for distance-only and 0.629 for EM. The EM-distance gap is only about 0.002 and should not be called significant without further evidence. Gene Spearman also falls. Depth and added RNA change together, so this table alone cannot isolate their causal contributions.

The paired dot plots share a zero-to-one mean core-cell IoU scale; they compare conditions, not time points. Expand the table for exact values and gene correlations. Fractional IoU is recorded separately. These are saved September 3 results, not a rerun after September 8 code fixes or a method comparison on the later 28 realism configurations.

---

## Slide 23 — Historical RL diagnosis

**Suggested time / 建议时间：1 minute**

### 中文

这一页的数字保留为历史对照，不是最新 donor-disjoint 数据上的 RL 实验。旧 frozen policy 的 REPLACE 有 6 个 corrected、16 个 damaged、34 个在 selected-cell GT 之外、2 个 owner map 未解决；20.7% 的选择是当时的最高 global-delta action。旧 entropy threshold=0.5 把 875 个非核 bins 全部放行。

它说明那个旧 policy 和旧 gate 没有带来预期改善，但不能据此判断最新数据上的 RL 上限。现在先校验数据和表达证据，不根据这些旧统计立即重新校准 entropy 或训练 PPO。

### English

These numbers remain as a historical comparison, not an RL experiment on the latest donor-disjoint dataset. The old frozen policy produced six corrected and sixteen damaged owners, with thirty-four outside selected-cell GT and two unresolved owner mappings. Only 20.7% of choices were the highest-global-delta action, and the old entropy threshold of 0.5 admitted all 875 scored non-nuclear bins.

That policy and gate did not deliver the intended improvement. This does not establish RL's potential on the latest data. We are checking the data and expression evidence first, rather than immediately recalibrating entropy or training PPO from these old statistics.

---

## Slide 24 — See the nonzero-support shortcut

**Suggested time / 建议时间：3 minutes**

### 中文

[先看三个图，再显示 pseudo coverage 边界，最后切换 depth-only。必要时换另一个 patch。]

这三个图来自同一区域的实际保存数据。左边是有 pseudo cell coverage 的 bins，中间是模拟数据中 counts 大于零的 bins，右边是真实 HD 中 counts 大于零的 bins。黑色只表示“有计数”，灰色表示没有载入的 barcode，不能当作零。图按 barcode array 坐标展示，不是 H&E 方向的截图。

[叠加轮廓。] 洋红色线表示 pseudo coverage 的并集边缘，不是每个 cell 的完整身份，也不是真实组织 GT。模拟的 nonzero 区域几乎直接描出这条边界，真实 HD 则在这些 pseudo masks 之外也有大量信号。这就是只用 counts>0 就能获得很高 support BA 的原因，不是已经恢复了每个细胞。

[切换 depth-only。] 去掉一些已有 counts，不能在空白区域创造缺失的表达。这里显示的是 seed 20260907 的单次矩阵；原来表中的 depth-only 0.9990 是三个 seeds 的均值。当前 patch 的 BA 单列，和全四个 contexts 的汇总分开。

四个 patch 都可以切换，默认按 ID 排序选第一个，没有为图挑最高分区域。这些区域已经用于诊断，不是新的 held-out validation。28 配置、78 次实现找到了数据问题，但没有重新跑方法排名，也没有证明某个 generator 已经足够真实。

### English

[Inspect the three maps, overlay pseudo coverage, then switch to depth-only. Optionally inspect another patch.]

These maps use actual saved data from the same region. Left: bins with any pseudo-cell coverage. Middle: nonzero pseudo counts. Right: nonzero real-HD counts. Black means counts greater than zero; gray means a barcode was not loaded, not a measured zero. Coordinates follow the barcode array, not H&E image orientation.

[Overlay the boundary.] Magenta outlines the union of pseudo-covered bins, not each cell identity and not real tissue GT. Nonzero pseudo expression nearly traces that edge, whereas real HD also has substantial signal outside it. This explains the high support balanced accuracy of a one-bit counts-greater-than-zero rule; it is not recovery of individual cells.

[Switch to depth-only.] Removing existing counts cannot create expression in empty regions. This view uses the saved seed-20260907 matrix; the original depth-only value of 0.9990 averages three seeds. The current context's BA is reported separately from the four-context summary.

All four patches are selectable, with the first sorted ID as the default rather than a best-looking selection. These are previously examined diagnostic regions, not another held-out validation. The 28 configurations and 78 realizations expose data problems, but do not establish a new method ranking or a realism-validated generator.

---

## Slide 25 — Compressed library-size variation

**Suggested time / 建议时间：2 minutes**

### 中文

[先看实际 Cycling T profiles 的累积分布，再按需要展开 CV 表。] 横轴是每个细胞的 panel UMI，按 log10(1+UMI) 排列；纵轴表示多少比例的细胞不超过该计数。粉色曲线在很窄的区间内陡升，说明模拟细胞的 UMI 更集中。曲线直接来自 579 个 generator scRNA cells 和 60 个模拟 profiles，不是根据均值/CV 拟合出来的示意分布。

这次确认的 scDesign3 问题不是所有 cell 的 mean UMI 都高，而是同类型细胞间的 library-size variation 被压缩。例如 Cycling T 的 mean UMI 接近，2933 对 2964，但 CV 从 0.829 降到 0.156；Pericytes 从 0.728 降到 0.082。展开表只展示 simulated cells 至少十个的类型。

从独立 generator scRNA 同类型抽取 library totals 的控制恢复了不少变异，但它还使用 membrane strength=0.25，不能把相对原始配方的所有变化都归于 library size。这也只修正一项统计，不代表 scRNA library 就是真实 HD RNA abundance。

等深度的 marker/type separation 结果并没有一致增强。因此我们不应该先把全部 marker 人为模糊，也不能把当前配置的问题归咎于 scDesign3 软件本身。

### English

[Read the actual Cycling T empirical distributions before opening the CV table.] The x-axis is panel UMI per cell, with log10(1+UMI) spacing. The y-axis is the fraction of cells at or below that count. The steep magenta rise means that simulated totals occupy a narrower range. These curves use 579 generator scRNA cells and 60 synthetic profiles, not a distribution inferred from means or CVs.

The confirmed scDesign3 issue is not that every mean UMI is inflated. It is compressed library-size variation among cells of the same type. Cycling T means are similar, 2,933 versus 2,964, but CV falls from 0.829 to 0.156. For pericytes, it falls from 0.728 to 0.082. The expandable table includes only types with at least ten simulated cells.

A control sampling type-matched library totals from independent generator scRNA restores much of the variability. It also uses membrane strength 0.25, so all changes relative to the original recipe cannot be attributed to library size. Restoring one statistic does not make scRNA library size a direct estimate of HD cellular RNA abundance.

Equal-depth type separation is not consistently stronger in simulation. We should not blur all markers by default or attribute a limitation of this type-only configuration to scDesign3 as a whole.

---

## Slide 26 — Unassigned tissue RNA is not all ambient

**Suggested time / 建议时间：2 minutes**

### 中文

局部四个 contexts 中，有 34.705% 的真实 counts 落在已建模 pseudo masks 外。旧流程把这些都作为 ambient 加入，但它们可能来自漏建模细胞、细胞突起、胞外表达和技术背景，来源不能混为一谈。

全片按 gene IDs 对齐 18,132 个 genes，检查了 11,222,500 个 raw bins。组织外的 counts 占全片 1.158%，但这不是组织内的 ambient rate，也不能直接替换上面的 34.705%。两者空间范围、分母和生物意义都不同。

组织外信号还随离组织距离变化：0–10 微米平均每 bin 6.006 UMI，超过 50 微米是 0.452。后续应把 modeled-cell RNA、unmodelled biological RNA 和 technical background 分开建模，不应任意选一个 background 百分比来压低工具分数。

### English

In the four local contexts, 34.705% of real counts lie outside the modelled pseudo masks. The old pipeline added this as ambient, but it may include unmodelled cells, processes, extracellular expression and technical background.

A whole-slide audit aligned the 18,132 genes by exact gene IDs and examined 11,222,500 raw bins. Off-tissue counts contribute 1.158% of the whole-slide panel total. This is not the in-tissue ambient rate and cannot replace 34.705%; the regions, denominators and biological interpretations differ.

Off-tissue signal also varies with distance: the mean is 6.006 UMIs per bin within ten micrometres of tissue and 0.452 beyond fifty micrometres. The next generator should distinguish modelled-cell RNA, unmodelled biological RNA and technical background rather than select an arbitrary fraction to reduce tool scores.

---

## Slide 27 — Shift observed nuclear bins over fixed GT

**Suggested time / 建议时间：2 minutes**

### 中文

[依次点击 0、2、4 µm；必要时切换 patch。]

这里保持 pseudo GT 不变，只移动提供给算法的 nuclear observations。移动的是 array column 方向上的核 bins，不是重新生成 ground truth。深色方块表示仍对应同一个 dominant owner；橙色叉号表示已经移到不同 owner 的位置。

当前视图给出该 patch 的 valid、不同 owner 和丢失数量。全四个 contexts 的不同 owner 比例分别是 0.53%、10.34%、28.86%，裁剪外丢失 0、46、94 个 bins。Baseline 不为零，是因为 fractional overlap 与 dominant-owner 定义不同。

这些是人为设置的 sensitivity controls，不说明真实 registration error 就是 2 或 4 µm。膜定位强度、核分配比例也只是受检验的假设，尚未生物校准。要从独立真实数据估计误差，而不是选择一个能让工具分数下降的噪声量。原始 shape prior 没有改。

### English

[Step through zero, two and four micrometres; optionally switch patch.]

Pseudo GT remains fixed while observed nuclear bins move along the barcode-array column direction. We do not regenerate the ground truth. Dark squares still map to the same dominant owner; orange crosses land on a different owner.

The visible readout reports valid, disagreeing and missing destinations for this patch. Across all four contexts, disagreement is 0.53%, 10.34% and 28.86%, with zero, 46 and 94 lost bins. Baseline disagreement is nonzero because fractional overlap and dominant-owner definitions differ.

These are imposed sensitivity controls, not evidence that real registration error is two or four micrometres. Membrane localization strength and nuclear allocation fractions are also assumptions under examination, not biologically calibrated values. Noise should be estimated independently rather than chosen to lower tool scores. The original shape prior is unchanged.

---

## Slide 28 — Make the data more realistic, not simply harder

**Suggested time / 建议时间：2 minutes**

### 中文

现在 donor 分离、新 patch 方法比较，以及 28 配置/78 次数据诊断都已经做过。下一步不是为了让工具掉分而加噪声，而是更接近真实 HD 的可测统计：完整 tissue support、cell-level variation、跨 2/8/16 微米的捕获与空间结构，以及独立的核观测误差。

没有某个已测试配置同时解决所有问题。加入 real replay 的数据还缺完整 cell-owner GT，不能直接当完全监督 benchmark。应先冻结来源可解释、统计经验证的生成设置，再换真正新的区域或 section，做同一组五方法比较；之后才考虑 entropy calibration 和新 PPO。

当前可以说推断与训练架构已跑通，表达项在某些 pseudo 条件下有效。但没有足够真实的最终 generator，也没有在最新数据上证明 RL 增益。工程测试通过和科学验证通过，是两件不同的事。

### English

Donor separation, the unseen-patch comparison and the 28-configuration, 78-realization data audit are already complete. The next step is not noise added merely to lower tool scores. It is agreement with measurable real-HD properties: full tissue support, cell-level variation, capture and spatial structure across two, eight and sixteen micrometres, and independent nuclear-observation error.

No tested configuration solves every mismatch. Real replay also lacks complete cell-owner GT and cannot become a fully supervised benchmark by assumption. We should freeze an interpretable, statistically checked generator, move to genuinely fresh regions or sections, and repeat the same five-method comparison. Entropy calibration and PPO come later.

The inference and training architecture works, and expression helps under some pseudo conditions. But there is no realism-validated final generator or demonstrated RL gain on the latest dataset. Passing engineering tests and passing scientific validation are different claims.

---

## Short Q&A notes / 简短问答备忘

### Why not use only image segmentation? / 为什么不只用图像分割？

Image boundaries constrain morphology, but they do not directly identify the cell that generated displaced or boundary RNA. The ownership model combines nuclear anchors, geometry, and expression.

图像边界约束 morphology，但不能直接回答 displaced RNA 或 boundary RNA 来自哪个 cell。Ownership model 需要同时使用 nuclear anchors、geometry 和 expression。

### Why keep empty nuclei? / 为什么保留 empty nuclei？

An empty nuclear count can result from sparse capture. The nucleus may still be a real physical cell with RNA in nearby non-nuclear bins. Nuclear QC should use segmentation and morphology rather than expression alone.

Nuclear count 为空可能只是 capture sparsity。这个 nucleus 仍可能是真实细胞，并在附近非核 bins 中有 RNA，因此不能只按 expression 过滤。

### Is the EM exact maximum-likelihood EM? / 这是严格的最大似然 EM 吗？

No. It is a generalized EM or alternating probabilistic soft-assignment algorithm that reuses the project's posterior-weighted likelihood and configurable spatial and expression weights.

不是。它是 generalized EM 或 alternating probabilistic soft assignment，复用了项目现有的 posterior-weighted likelihood 以及可配置的 spatial 和 expression weights。

### Why not add EM probability to the RL reward? / 为什么不把 EM probability 加进 RL reward？

EM and the global objective already use related expression and distance evidence. Adding EM probability again would double-count that evidence and could train RL to imitate EM rather than correct it.

EM 和 global objective 已经使用了相关的 expression 与 distance evidence。再次加入 EM probability 会重复计分，并可能让 RL 只模仿 EM，而不是纠正 EM。

### Does the current result show that RL is unnecessary? / 当前结果是否说明 RL 没有意义？

No. It shows that the historical frozen policy and its calibration did not improve the older EM benchmark. RL could still help on genuinely ambiguous boundaries, but that requires a realistic benchmark, a selective entropy gate, and a better-aligned objective and policy.

不能这样下结论。历史结果只说明当时的 frozen policy 和 calibration 没有改善旧 EM benchmark；最新数据没有重跑 RL。RL 仍可能在真正 ambiguous 的 boundary 上有用，但需要更真实的 benchmark、更有选择性的 entropy gate，以及更一致的 objective 和 policy。

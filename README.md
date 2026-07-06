# spa-ds (originally qsi-tk)

 Data science toolkit for spectroscopic profiling signals/data.

# Installation

> pip install spa-ds

# Contents

This package is a master library containing various previous packages published by our team.

<table>
    <tbody>
        <tr>
            <td>module</td>
            <td>sub-module</td>
            <td>description</td>
            <td>standalone pypi package</td>
            <td>publication</td>
        </tr>
        <tr>
            <td colspan = 1 rowspan = 3>spa.io</td>
            <td>spa.io.load</td>
            <td>File I/O, Dataset loading</td>
            <td></td>
            <td>Provides 40+ open datasets. 15+ with publications</td>
        </tr>
        <tr>
            <td colspan = 1>spa.io.aug</td>
            <td>Data augmentation, e.g., generative models</td>
            <td></td>
            <td>Data aug with deep generative models. e.g., " variational autoencoders, generative adversarial networks, autoregressive models, KDE, normalizing flow models, energy-based models, and score-based models. "</td>
        </tr>
        <tr>
            <td>spa.io.pre</td>
            <td>Data preprocessing, e.g., window filter, x-binning, baseline removal.</td>
            <td></td>
            <td>Enhanced data preprocessing with novel window function in Raman spectroscopy: Leveraging feature selection and machine learning for raspberry origin identification [J]. Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy. 2024. doi: 10.1016/j.saa.2024.124913</td>
        </tr>
        <tr>
            <td colspan = 2>spa.vis</td>
            <td>Plotting</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td colspan = 2>spa.cs</td>
            <td>compressed sensing</td>
            <td>cs1</td>
            <td>Adaptive compressed sensing of Raman spectroscopic profiling data for discriminative tasks [J]. Talanta, 2020, doi: 10.1016/j.talanta.2019.120681
            <br/>
            Task-adaptive eigenvector-based projection (EBP) transform for compressed sensing: A case study of spectroscopic profiling sensor [J]. Analytical Science Advances. Chemistry Europe, 2021, doi: 10.1002/ansa.202100018
            <br/>
            Compressed Sensing library for spectroscopic profiling data [J]. Software Impacts, 2023, doi: 10.1016/j.simpa.2023.100492
            <br/>
            Secured telemetry based on time-variant sensing matrix – An empirical study of spectroscopic profiling, Smart Agricultural Technology, Volume 5, 2023, doi: 10.1016/j.atech.2023.100268
            <br/>
            Variational Auto-Encoder based Deep Compressed Sensing on Raman Spectroscopy [J]. Smart Agricultural Technology. 2025
            </td>
        </tr>
        <tr>
            <td colspan = 1 rowspan = 4>spa.fs</td>
        </tr>
        <tr>
            <td>spa.fs.nch_time_series_fs</td>
            <td>channel alignment for e-nose; multi-channel e-nose/e-tongue data fs with 1d-laplacian conv kernel</td>
            <td></td>
            <td>基于电子鼻和一维拉普拉斯卷积核的奶粉基粉产地鉴别,2024,doi: 10.13982/j.mfst.1673-9078.2024.5.0299
            </td>
        </tr>
        <tr>
            <td>spa.fs.glasso</td>
            <td>Structured-fs of Raman data with group lasso</td>
            <td />
            <td>Cheese brand identification with Raman spectroscopy and sparse group LASSO [J], Journal of Food Composition and Analysis, 2025, doi: 10.1016/j.jfca.2025.107371</td>
        </tr>
        <tr>
            <td>spa.fs.mt</td>
            <td>Multi-task feature selection for yogurt fermentation analysis</td>
            <td />
            <td>Studying yogurt fermentation dynamics using multi-task feature selection, 2025, 2nd-round review</td>
        </tr>
        <tr>
            <td rowspan = 2>spa.kernel</td>
            <td>spa.kernel.*</td>
            <td>Implementation of 31 atom kernel types</td>
            <td>ackl</td>
            <td>Analytical chemistry kernel library for spectroscopic profiling data, Food Chemistry Advances, Volume 3, 2023, 100342, ISSN 2772-753X, https://doi.org/10.1016/j.focha.2023.100342.</td>
        </tr>
        <tr>
            <td>spa.kernel.mkl</td>
            <td>Multi-kernel learning; PSO-MKL, GA-MKL</td>
            <td></td>
            <td>In progress</td>
        </tr>
        <tr>
            <td rowspan = 2>spa.dr</td>
            <td>spa.dr.metrics</td>
            <td>Dimensionality Reduction (DR) quality metrics</td>
            <td>pyDRMetrics, wDRMetrics</td>
            <td>pyDRMetrics - A Python toolkit for dimensionality reduction quality assessment, Heliyon, Volume 7, Issue 2, 2021, e06199, ISSN 2405-8440, doi: 10.1016/j.heliyon.2021.e06199.</td>
        </tr>
        <tr>
            <td>spa.dr.mf</td>
            <td>matrix-factorization based DR</td>
            <td>pyMFDR</td>
            <td>Matrix Factorization Based Dimensionality Reduction Algorithms - A Comparative Study on Spectroscopic Profiling Data [J], Analytical Chemistry, 2022. doi: 10.1021/acs.analchem.2c01922</td>
        </tr>
        <tr>
            <td rowspan = 4>spa.cla</td>
            <td>spa.cla.metrics</td>
            <td>classifiability analysis</td>
            <td>pyCLAMs, wCLAMs</td>
            <td>A unified classifiability analysis framework based on meta-learner and its application in spectroscopic profiling data [J]. Applied Intelligence, 2021, doi: 10.1007/s10489-021-02810-8
            <br/> 
            pyCLAMs: An integrated Python toolkit for classifiability analysis [J]. SoftwareX, 2022, doi: 10.1016/j.softx.2022.101007</td>
        </tr>
        <tr>
            <td>spa.cla.ensemble</td>
            <td>homo-stacking, hetero-stacking, FSSE</td>
            <td rowspan = 3>pyNNRW</td>
            <td rowspan = 3>Spectroscopic Profiling-based Geographic Herb Identification by Neural Network with Random Weights [J]. Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy, 2022, doi: 10.1016/j.saa.2022.121348
            <br/>
            Geographical origin identification of dendrobium officinale based on NNRW-stacking ensembles. Machine Learning with Applications [J]. 2024. doi: 10.1016/j.mlwa.2024.100594
            </td>
        </tr>
        <tr>
            <td>spa.cla.kernel</td>
            <td>kernel-NNRW</td>
        </tr>
        <tr>
            <td>spa.cla.nnrw</td>
            <td>neural networks with random weights</td>
        </tr>
        <tr>
            <td rowspan = 1>spa.regress</td>
            <td></td>
            <td>Regression algorithms, e.g., GW-KNNR (Gaussian-weighted K-nearest neighbor regressor).</td>
            <td></td>
            <td>Quantification of Cow Milk in Adulterated Goat Milk Using Raman Spectroscopy and Machine Learning[J]. Microchemical Journal, 2025, doi: 10.1016/j.microc.2025.114319</td>
        </tr>
        <tr>
            <td rowspan = 1>spa.pipeline</td>
            <td></td>
            <td>General data analysis pipelines.</td>
            <td></td>
            <td>
            Building an Information Infrastructure of Spectroscopic Profiling Data for Food-Drug Quality and Safety Management [J]. Enterprise Information Systems, 2019, doi: 10.1080/17517575.2019
            <br/>
            Machine learning-assisted MALDI-TOF MS toward rapid classification of milk products[J]. Journal of Dairy Science, 2024, doi:10.3168/jds.2024-24886</td>
        </tr>
        <tr>
            <td rowspan = 1>spa.gui</td>
            <td></td>
            <td>Web-based apps. e.g., `python -m spa.gui.chaihu` will launch the app for bupleurum origin discrimination.</td>
            <td></td>
            <td>Rapid Raman Spectroscopy Analysis Assisted with Machine Learning: A Case Study on Radix Bupleuri[J], Journal of the Science of Food and Agriculture, 2024. doi:10.1002/jsfa.14012</td>
        </tr>
    </tbody>
</table>



 刚提的两项设置（我上一条只做了一半）
恢复默认：Service 方法已加，但还没接 API 端点和前端按钮。
语言/画幅等限定下拉：未做，仍是自由文本。

B. 产品/平台层——大块完全没做
Export Presets（抖音/YouTube/B站 多尺寸批量导出）——未做。
Safety & Review / 内容审核——未做。
Model Routing 未接线：Settings 里 ScriptModel/ImageModel/VideoModel/VoiceModel 字段能存，但生成时并没真正按它路由到不同模型（provider 仍读 env/MediaOptions）。

C. 引擎模块——首版已跑通，深度未做
③ Generation Router：算出并记录路线，但没有真正差异化的生成路径（多关键帧/首帧/图生视频只是标签，未落到不同流程）。
④ Reference Director：只有 prompt 优先级 + 禁止项文本；没有参考图 embedding / IP-Adapter 传图做人脸强一致。
⑤ Lineart/专业分镜：结构字段有了，没有真正的线稿渲染 / ControlNet 结构锁定。
⑥ QC Gate：规则检查，没有真实画面漂移检测（需视觉模型）。
⑦ Repair Path：映射到重生成策略，没有局部重绘/局部重算。
⑧ Dependency Graph：能看角色→镜头引用，没有"改角色→自动批量重算受影响镜头"的传播。
⑩ Experiment Run：会记录，没有 A/B 实验编排 / ReRank 训练集导出。
⑪ Delivery Package：MP4+BGM+SRT+manifest，缺封面帧、工程包(zip)、审阅态、资产清单打包。
D. 皇冠层——规则引擎已落地，其余未做
Director RAG（影视教材/摄影/剪辑知识检索）——未做。
真实 Video Critic 模型——当前是启发式占位打分，非真实模型。
E. 生成层仍是占位/默认省成本
视频：默认走 FFmpeg 伪运镜；真 text2video（Wan2.2）已接但默认关。
图像一致性：只靠 prompt+seed，无真正 img2img/人脸锁定（默认）。
mflux 本地图像：因磁盘不足未启用；Ollama 本地 LLM 非默认（现用 DeepSeek）。
Cost Estimator：给的是调用次数，不是真实金额（无单价表）。
一句话
11 个引擎模块 + 皇冠规则引擎都有可运行首版；没做的主要是：产品化平台层（鉴权/权限/支付/存储/发布集成/审核）+ 各引擎的"真模型/真渲染"深度 + Director RAG + 真实 Critic 训练。
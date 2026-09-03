---
layout: page
title: Research
permalink: /research/
order: 1
---

My current work at Adobe Research is directed towards *audio language modeling and evaluations for full-duplex speech systems*, reinforcement learning for LLM agents, and multimodal omni models. Previously, I have worked on *agentic AI* and *multimodal language understanding* — spanning audio and speech, documents, and language.


During my Ph.D. at the University of Maryland, my work centered on multimodal document understanding, information extraction, and long-context modeling. 
[Google Scholar]({{ site.scholar_url }})

## Audio &amp; Speech: Full-Duplex Language Modeling

I am currently building audio language models for full-duplex speech interaction — systems that can listen and generate speech simultaneously, handle interruptions, backchannels, and turn-taking natively, rather than the rigid turn-based latency of today's half-duplex voice assistants. This includes benchmarking document grounding, hallucination, and instruction-following in full-duplex voice agents, reasoning-and-retrieval-while-speaking architectures, and efficient parametric memory for omni language models. Papers and project pages will be linked here as they become public.

<ul class="pub-list">
<li>
<span class="pub-title">Duplex-R1: Full-Duplex Audio LMs that Reason, Retrieve, and Speak While Searching</span>
<span class="pub-authors"><i>Puneet Mathur</i>, Nedim Lipka, Zeyu Jin, Dinesh Manocha</span>
<span class="pub-venue">In submission, 2026 &middot; <span class="pub-soon">(Coming soon)</span></span>
</li>
<li>
<span class="pub-title"><a href="https://drive.google.com/file/d/19PFdT7quVGvLRk3se8uoPB6xmPqB4tyJ/view?usp=sharing">DuplexSpeechBench&ndash;IFEval: Evaluating Implicit Instruction Following in Full-Duplex Voice Agents</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Dinesh Manocha</span>
<span class="pub-links"><a href="http://dsb-ifeval.github.io/">Project Page</a></span>
<span class="pub-venue">In submission, 2026</span>
</li>
<li>
<span class="pub-title"><a href="https://drive.google.com/file/d/1AXy6N1eUe5DbDj20cRdw9fHPKlyASnVm/view">DuplexSpeechBench&ndash;Document Grounding: Benchmarking Document Grounding and Hallucinations in Full-Duplex Voice Agents</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Nedim Lipka, Zeyu Jin, Dinesh Manocha</span>
<span class="pub-venue">In submission, 2026<span class="pub-soon"></span></span>
</li>
<li>
<span class="pub-title"><a href="https://arxiv.org/pdf/2608.09227">Omni2LoRA: Coherence-Preserving Parametric Memory for Efficient Omni Language Models</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Manan Suri, Dinesh Manocha</span>
<span class="pub-links"><a href="http://omni2lora.github.io/">Project Page</a></span>
<span class="pub-venue">In submission, 2026</span>
</li>
</ul>

This builds on earlier work of mine at the intersection of speech, retrieval, and language modeling:

<ul class="pub-list">
<li>
<span class="pub-title"><a href="https://aclanthology.org/2023.findings-emnlp.757.pdf">PersonaLM: Language Model Personalization via Domain-distributed Span Aggregated K-Nearest N-gram Retrieval Augmentation</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Zhe Liu, Ke Li, Yingyi Ma, Gil Keren, Zeeshan Ahmed, Dinesh Manocha, Xuedong Zhang</span>
<span class="pub-venue">EMNLP 2023 (Findings)</span>
</li>
<li>
<span class="pub-title"><a href="https://aclanthology.org/2024.lrec-main.457.pdf">DOC-RAG: ASR Language Model Personalization with Domain-Distributed Co-occurrence Retrieval Augmentation</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Zhe Liu, Ke Li, Yingyi Ma, Gil Keren, Zeeshan Ahmed, Dinesh Manocha, Xuedong Zhang</span>
<span class="pub-venue">COLING 2024</span>
</li>
<li>
<span class="pub-title"><a href="https://www.isca-speech.org/archive/pdfs/interspeech_2022/mathur22_interspeech.pdf">DocLayoutTTS: Dataset and Baselines for Layout-informed Document-level Neural Speech Synthesis</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Franck Dernoncourt, Quan Hung Tran, Jiuxiang Gu, Ani Nenkova, Vlad Morariu, Rajiv Jain, Dinesh Manocha</span>
<span class="pub-venue">Interspeech 2022</span>
</li>
<li>
<span class="pub-title"><a href="https://isca-speech.org/archive/Interspeech_2020/pdfs/2649.pdf">Risk Forecasting from Earnings Calls Acoustics and Network Correlations</a></span>
<span class="pub-authors">Ramit Sawhney, Arshiya Aggarwal, Piyush Khanna, <i>Puneet Mathur</i>, Taru Jain, Rajiv Ratn Shah</span>
<span class="pub-venue">Interspeech 2020</span>
</li>
<li>
<span class="pub-title"><a href="https://www.aclweb.org/anthology/2020.emnlp-main.643.pdf">VolTAGE: Volatility Forecasting via Text Audio Fusion with Graph Convolution Networks for Earnings Calls</a></span>
<span class="pub-authors">Ramit Sawhney, Arshiya Aggarwal, Piyush Khanna, Taru Jain, <i>Puneet Mathur</i>, Rajiv Ratn Shah</span>
<span class="pub-venue">EMNLP 2020</span>
</li>
<li>
<span class="pub-title"><a href="https://ieeexplore.ieee.org/document/9414373">Meta-learning for Low-Resource Speech Emotion Recognition</a></span>
<span class="pub-authors">Suransh Chopra*, <i>Puneet Mathur*</i>, Ramit Sawhney, Rajiv Ratn Shah</span>
<span class="pub-venue">ICASSP 2021</span>
</li>
</ul>

## Agentic AI, LLM Reasoning &amp; Reinforcement Learning

<ul class="pub-list">
<li>
<span class="pub-title"><a href="https://arxiv.org/abs/2511.08798">ARGUS: Structured Uncertainty guided Clarification for LLM Agents</a></span>
<span class="pub-authors">Manan Suri, <i>Puneet Mathur</i>, Nedim Lipka, Franck Dernoncourt, Ryan A. Rossi, Dinesh Manocha</span>
<span class="pub-venue">ACL 2026</span>
</li>
<li>
<span class="pub-title"><a href="https://arxiv.org/pdf/2603.06138">Partial Policy Gradients for RL in LLMs</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Branislav Kveton, Subhojyoti Mukherjee, Viet Dac Lai</span>
<span class="pub-venue">In submission, 2026</span>
</li>
<li>
<span class="pub-title">Cluster-R1: Large Reasoning Models Are Instruction-following Clustering Agents</span>
<span class="pub-authors">Peijun Qing, <i>Puneet Mathur</i>, Nedim Lipka, Varun Manjunatha, Ryan A. Rossi, Franck Dernoncourt, Saeed Hassanpour, Soroush Vosoughi</span>
<span class="pub-venue">In submission, 2026</span>
</li>
</ul>

## Multimodal Document Intelligence, Attribution &amp; RAG

<ul class="pub-list">
<li>
<span class="pub-title"><a href="https://research.adobe.com/publication/charteval-llm-driven-chart-generation-evaluation-using-scene-graph-parsing/">ChartEval: LLM-Driven Chart Generation Evaluation Using Scene Graph Parsing</a></span>
<span class="pub-authors">Kanika Goswami, <i>Puneet Mathur</i>, Franck Dernoncourt, Ryan A. Rossi, Vivek Gupta, Dinesh Manocha</span>
<span class="pub-venue">AACL 2025</span>
</li>
<li>
<span class="pub-title"><a href="https://arxiv.org/abs/2505.19360">ChartLens: Fine-grained Visual Attribution in Charts</a></span>
<span class="pub-authors">Manan Suri, <i>Puneet Mathur</i>, Nedim Lipka, Franck Dernoncourt, Ryan A. Rossi, Dinesh Manocha</span>
<span class="pub-venue">ACL 2025</span>
</li>
<li>
<span class="pub-title"><a href="https://arxiv.org/abs/2506.01344">Follow the Flow: Fine-grained Flowchart Attribution with Neurosymbolic Agents</a></span>
<span class="pub-authors">Manan Suri, <i>Puneet Mathur</i>, Nedim Lipka, Franck Dernoncourt, Ryan A. Rossi, Vivek Gupta, Dinesh Manocha</span>
<span class="pub-venue">EMNLP 2025</span>
</li>
<li>
<span class="pub-title"><a href="https://aclanthology.org/2025.emnlp-main.1585.pdf">DELOC: Document Element Localizer</a></span>
<span class="pub-authors">Hammad Ayyubi, <i>Puneet Mathur</i>, Mehrab Tanjim, Vlad I Morariu</span>
<span class="pub-venue">EMNLP 2025</span>
</li>
<li>
<span class="pub-title"><a href="https://aclanthology.org/2025.findings-emnlp.81/">SQLSpace: A Representation Space for Text-to-SQL to Discover and Mitigate Robustness Gaps</a></span>
<span class="pub-authors">Neha Srikanth, Victor Bursztyn, <i>Puneet Mathur</i>, Ani Nenkova</span>
<span class="pub-venue">EMNLP 2025 (Findings)</span>
</li>
<li>
<span class="pub-title"><a href="https://dl.acm.org/doi/pdf/10.1145/3706599.3719846">DocVoyager: Anticipating Users' Information Needs and Guiding Document Reading through Question Answering</a></span>
<span class="pub-authors">Yoonjoo Lee, Nedim Lipka, Zichao Wang, Ryan Rossi, <i>Puneet Mathur</i>, Tong Sun, Alexa Siu</span>
<span class="pub-venue">CHI 2025</span>
</li>
<li>
<span class="pub-title"><a href="https://aclanthology.org/2025.naacl-long.310/">VisDoM: Multi-Document QA with Visually Rich Elements Using Multimodal Retrieval-Augmented Generation</a></span>
<span class="pub-authors">Manan Suri, <i>Puneet Mathur</i>, Franck Dernoncourt, Kanika Goswami, Ryan A. Rossi, Dinesh Manocha</span>
<span class="pub-venue">NAACL 2025</span>
</li>
<li>
<span class="pub-title"><a href="https://arxiv.org/abs/2502.00322">MoDS: Moderating a Mixture of Document Speakers to Summarize Debatable Queries in Document Collections</a></span>
<span class="pub-authors">Nishant Balepur, Alexa Siu, Nedim Lipka, Franck Dernoncourt, Tong Sun, Jordan Lee Boyd-Graber, <i>Puneet Mathur</i></span>
<span class="pub-venue">NAACL 2025</span>
</li>
<li>
<span class="pub-title"><a href="https://aclanthology.org/2024.emnlp-demo.26.pdf">MATSA: Multi-Agent Table Structure Attribution</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Alexa Siu, Nedim Lipka, Tong Sun</span>
<span class="pub-venue">EMNLP 2024</span>
</li>
<li>
<span class="pub-title"><a href="https://aclanthology.org/2024.emnlp-main.867.pdf">DocEdit-v2: Document Structure Editing Via Multimodal LLM Grounding</a></span>
<span class="pub-authors">Manan Suri, <i>Puneet Mathur</i>, Franck Dernoncourt, Rajiv Jain, Vlad I Morariu, Ramit Sawhney, Preslav Nakov, Dinesh Manocha</span>
<span class="pub-venue">EMNLP 2024</span>
</li>
<li>
<span class="pub-title"><a href="https://openreview.net/forum?id=1ba209BACA#discussion">Agent-DocEdit: Language-Instructed LLM Agent for Content-Rich Document Editing</a></span>
<span class="pub-authors">Te-Lin Wu, Rajiv Jain, Yufan Zhou, <i>Puneet Mathur</i>, Vlad I Morariu</span>
<span class="pub-venue">COLM 2024</span>
</li>
<li>
<span class="pub-title"><a href="https://aclanthology.org/2024.acl-demos.22.pdf">DocPilot: Copilot for Automating PDF Edit Workflows in Documents</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Alexa Siu, Varun Manjunatha, Tong Sun</span>
<span class="pub-venue">ACL 2024</span>
</li>
<li>
<span class="pub-title"><a href="https://aclanthology.org/2024.lrec-main.458.pdf">DocScript: Document-level Script Event Prediction</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Rajiv Jain, Vlad Morariu, Aparna Garimella, Franck Dernoncourt, Jiuxiang Gu, Ramit Sawhney, Preslav Nakov, Dinesh Manocha</span>
<span class="pub-venue">COLING 2024</span>
</li>
<li>
<span class="pub-title"><a href="https://ojs.aaai.org/index.php/AAAI/article/view/25282/25054">DocEdit: Language-guided Document Editing</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Rajiv Jain, Jiuxiang Gu, Franck Dernoncourt, Dinesh Manocha, Vlad Morariu</span>
<span class="pub-venue">AAAI 2023</span>
</li>
<li>
<span class="pub-title"><a href="https://openaccess.thecvf.com/content/WACV2023/papers/Mathur_LayerDoc_Layer-Wise_Extraction_of_Spatial_Hierarchical_Structure_in_Visually-Rich_Documents_WACV_2023_paper.pdf">LayerDoc: Layer-wise Extraction of Spatial Hierarchical Structure in Visually-Rich Documents</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Rajiv Jain, Ashutosh Mehra, Jiuxiang Gu, Franck Dernoncourt, Anandhavelu N, Quan Tran, Verena Kaynig-Fittkau, Ani Nenkova, Dinesh Manocha, Vlad Morariu</span>
<span class="pub-venue">WACV 2023</span>
</li>
<li>
<span class="pub-title"><a href="https://aclanthology.org/2022.emnlp-main.51">DocInfer: Document-level Natural Language Inference using Optimal Evidence Selection</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Gautam Kunapuli, Riyaz Ahmad Bhat, Manish Shrivastava, Dinesh Manocha, Maneesh Singh</span>
<span class="pub-venue">EMNLP 2022</span>
</li>
<li>
<span class="pub-title"><a href="https://aclanthology.org/2022.findings-emnlp.139">DocFin: Multimodal Financial Prediction and Bias Mitigation using Semi-structured Documents</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Mihir Goyal, Ramit Sawhney, Ritik Mathur, Jochen L. Leidner, Franck Dernoncourt, Dinesh Manocha</span>
<span class="pub-venue">EMNLP 2022 (Findings)</span>
</li>
<li>
<span class="pub-title"><a href="https://aclanthology.org/2022.naacl-main.73/">DocTime: A Document-level Temporal Dependency Graph Parser</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Vlad I Morariu, Verena Kaynig-Fittkau, Jiuxiang Gu, Franck Dernoncourt, Quan Hung Tran, Ani Nenkova, Dinesh Manocha, Rajiv Jain</span>
<span class="pub-venue">NAACL 2022</span>
</li>
<li>
<span class="pub-title"><a href="https://aclanthology.org/2021.acl-short.67/">TIMERS: Document-level Temporal Relation Extraction</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Rajiv Jain, Franck Dernoncourt, Vlad Morariu, Quan Hung Tran, Dinesh Manocha</span>
<span class="pub-venue">ACL 2021</span>
</li>
</ul>

## Multimodal AI, Affective Computing &amp; Video Understanding

<ul class="pub-list">
<li>
<span class="pub-title"><a href="https://aclanthology.org/2024.lrec-main.1244.pdf">Saliency-Aware Interpolative Augmentation for Multimodal Financial Prediction</a></span>
<span class="pub-authors">Samyak Jain, Parth Chhabra, Atula Neerkaje, <i>Puneet Mathur</i>, Ramit Sawhney, Shivam Agarwal, Preslav Nakov, Sudheer Chava, Dinesh Manocha</span>
<span class="pub-venue">COLING 2024</span>
</li>
<li>
<span class="pub-title"><a href="https://dl.acm.org/doi/pdf/10.1145/3503161.3548380">MONOPOLY: Financial Prediction from MONetary POLicY Conference Videos Using Multimodal Cues</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Atula Tejaswi Neerkaje, Malika Chhibber, Ramit Sawhney, Fu-Ming Guo, Franck Dernoncourt, Sanghamitra Dutta, Dinesh Manocha</span>
<span class="pub-venue">ACM Multimedia 2022</span>
</li>
<li>
<span class="pub-title"><a href="https://arxiv.org/abs/2203.14456">3MASSIV: Multilingual, Multimodal and Multi-Aspect Dataset of Social Media Short Videos</a></span>
<span class="pub-authors">Vikram Gupta, Trisha Mittal, <i>Puneet Mathur</i>, Vaibhav Mishra, Mayank Maheshwari, Aniket Bera, Debdoot Mukherjee, Dinesh Manocha</span>
<span class="pub-links"><a href="https://sharechat.com/research/3massiv">Dataset</a></span>
<span class="pub-venue">CVPR 2022</span>
</li>
<li>
<span class="pub-title"><a href="https://aclanthology.org/2021.acl-long.526/">Multimodal Multi-Speaker Merger and Acquisition (M3A) Financial Forecasting: A New Task, Dataset, and Neural Baselines</a></span>
<span class="pub-authors">Ramit Sawhney, Mihir Goyal, Prakhar Goel, <i>Puneet Mathur</i>, Rajiv Ratn Shah</span>
<span class="pub-venue">ACL 2021</span>
</li>
<li>
<span class="pub-title"><a href="https://arxiv.org/pdf/2103.06541.pdf">Affect2MM: Affective Analysis of Multimedia Content Using Emotion Causality</a></span>
<span class="pub-authors">Trisha Mittal, <i>Puneet Mathur</i>, Aniket Bera, Dinesh Manocha</span>
<span class="pub-venue">CVPR 2021</span>
</li>
<li>
<span class="pub-title"><a href="https://aclanthology.org/2021.naacl-main.387/">Multitask Learning for Emotionally Analyzing Sexual Abuse Disclosures</a></span>
<span class="pub-authors">Ramit Sawhney, <i>Puneet Mathur</i>, Taru Jain, Akash Kumar Gautam, Rajiv Ratn Shah</span>
<span class="pub-venue">NAACL 2021</span>
</li>
<li>
<span class="pub-title"><a href="https://arxiv.org/abs/2102.11922">Dynamic Graph Modeling of Simultaneous EEG and Eye-tracking Data For Reading Task Identification</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Trisha Mittal, Dinesh Manocha</span>
<span class="pub-venue">ICASSP 2021</span>
</li>
<li>
<span class="pub-title"><a href="https://dl.acm.org/doi/10.1145/3394171.3413752">Multimodal Multitask Financial Risk Forecasting</a></span>
<span class="pub-authors">Ramit Sawhney, <i>Puneet Mathur</i>, Piyush Khanna, Ayush Mangal, Rajiv Ratn Shah</span>
<span class="pub-venue">ACM Multimedia 2020 <span class="pub-flag">(Oral)</span></span>
</li>
</ul>

## Early Research: Vision, NLP &amp; Social Media (2018&ndash;2020)

<ul class="pub-list">
<li>
<span class="pub-title"><a href="https://ieeexplore.ieee.org/document/9054672">Mixup Multi-Attention Multi-Tasking Model for Early-Stage Leukemia Identification</a></span>
<span class="pub-authors"><i>Puneet Mathur*</i>, Mehak Piplani*, Ramit Sawhney, Rajiv Ratn Shah</span>
<span class="pub-venue">ICASSP 2020</span>
</li>
<li>
<span class="pub-title"><a href="https://ieeexplore.ieee.org/document/9053177">Rethinking Retinal Landmark Localization as Pose Estimation: Naive Single Stacked Network for Optic Disk and Fovea Detection</a></span>
<span class="pub-authors">Shishira Maiya*, <i>Puneet Mathur*</i></span>
<span class="pub-venue">ICASSP 2020</span>
</li>
<li>
<span class="pub-title"><a href="https://link.springer.com/chapter/10.1007/978-3-030-45442-5_33">Utilizing Temporal Psycholinguistic Cues for Suicidal Intent Estimation</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Ramit Sawhney, Shivang Chopra, Rajiv Ratn Shah</span>
<span class="pub-venue">ECIR 2020 (Short)</span>
</li>
<li>
<span class="pub-title"><a href="https://www.aaai.org/ojs/index.php/ICWSM/article/view/7292">#MeTooMA: Multi-Aspect Annotations of Tweets Related to the MeToo Movement</a></span>
<span class="pub-authors">Akash Gautam*, <i>Puneet Mathur*</i>, Rakesh Gosangi, Debanjan Mahata, Ramit Sawhney, Rajiv Ratn Shah</span>
<span class="pub-links"><a href="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JN4EYU">Dataset</a></span>
<span class="pub-venue">ICWSM 2020</span>
</li>
<li>
<span class="pub-title"><a href="https://www.aaai.org/ojs/index.php/AAAI/article/view/5374">Hindi-English Hate Speech Detection: Author Profiling, Debiasing, and Practical Perspectives</a></span>
<span class="pub-authors">Shivang Chopra, Ramit Sawhney, <i>Puneet Mathur</i>, Rajiv Ratn Shah</span>
<span class="pub-venue">AAAI 2020 <span class="pub-flag">(Oral)</span></span>
</li>
<li>
<span class="pub-title"><a href="https://ieeexplore.ieee.org/document/8658665">Exploring Classification of Histological Disease Biomarkers from Renal Biopsy Images</a></span>
<span class="pub-authors"><i>Puneet Mathur*</i>, Meghna P. Ayyar*, Rajiv Ratn Shah, Shree G Sharma</span>
<span class="pub-links"><a href="{{ site.baseurl }}/assets/wacv_paper.pdf">Poster</a> &middot; <a href="https://www.youtube.com/watch?v=iuas8M8jTkg&t=2s">Video</a></span>
<span class="pub-venue">WACV 2019</span>
</li>
</ul>

## Workshop Papers

<ul class="pub-list">
<li>
<span class="pub-title">Suicide Risk Assessment via Temporal Psycholinguistic Modeling</span>
<span class="pub-authors"><i>Puneet Mathur</i>, Ramit Sawhney, Rajiv Ratn Shah</span>
<span class="pub-venue">AAAI Student Abstract and Poster 2020</span>
</li>
<li>
<span class="pub-title">An Iterative Approach for Identifying Complaint Based Tweets in Social Media Platforms</span>
<span class="pub-authors">Gyanesh Anand, Akash Kumar Gautam, <i>Puneet Mathur</i>, Debanjan Mahata, Rajiv Ratn Shah, Ramit Sawhney</span>
<span class="pub-venue">AAAI Student Abstract and Poster 2020</span>
</li>
<li>
<span class="pub-title"><a href="https://www.aclweb.org/anthology/N19-3019">SNAP-BATNET: Cascading Author Profiling and Social Network Graphs for Suicide Ideation Detection on Social Media</a></span>
<span class="pub-authors">Rohan Mishra, Pradyumna Prakhar Sinha, Ramit Sawhney, Debanjan Mahata, <i>Puneet Mathur</i>, Rajiv Ratn Shah</span>
<span class="pub-venue">NAACL Student Research Workshop 2019</span>
</li>
<li>
<span class="pub-title"><a href="https://www.aclweb.org/anthology/N19-3018">Speak Up, Fight Back! Detection of Social Media Disclosures of Sexual Harassment</a></span>
<span class="pub-authors">Arijit Ghosh Chowdhury, Ramit Sawhney, <i>Puneet Mathur</i>, Rajiv Ratn Shah</span>
<span class="pub-venue">NAACL Student Research Workshop 2019</span>
</li>
<li>
<span class="pub-title"><a href="http://aclweb.org/anthology/W18-5907">Identification of Emergency Blood Donation Request on Twitter</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Meghna Ayyar, Sahil Chopra, Simra Shahid, Laiba Mehnaz, Rajiv Shah</span>
<span class="pub-links"><a href="{{ site.baseurl }}/assets/poster_smm4h.pdf">Poster</a></span>
<span class="pub-venue">SMM4H Workshop, EMNLP 2018</span>
</li>
<li>
<span class="pub-title"><a href="http://aclweb.org/anthology/W18-5118">Did You Offend Me? Classification of Offensive Tweets in Hinglish Language</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Ramit Sawhney, Meghna Ayyar, Rajiv Shah</span>
<span class="pub-links"><a href="{{ site.baseurl }}/assets/ALW_poster.pdf">Poster</a></span>
<span class="pub-venue">Abusive Language Workshop (ALW2), EMNLP 2018</span>
</li>
<li>
<span class="pub-title"><a href="http://www.aclweb.org/anthology/W18-6223">Exploring and Learning Suicidal Ideation Connotations on Social Media with Deep Learning</a></span>
<span class="pub-authors">Ramit Sawhney, Prachi Manchanda, <i>Puneet Mathur</i>, Rajiv Shah, Raj Singh</span>
<span class="pub-links"><a href="{{ site.baseurl }}/assets/WASSA-2018.pdf">Poster</a></span>
<span class="pub-venue">WASSA Workshop, EMNLP 2018</span>
</li>
<li>
<span class="pub-title"><a href="http://www.aclweb.org/anthology/W18-3504">Detecting Offensive Tweets in Hindi-English Code-Switched Language</a></span>
<span class="pub-authors"><i>Puneet Mathur</i>, Rajiv Shah, Ramit Sawhney, Debanjan Mahata</span>
<span class="pub-links"><a href="https://www.youtube.com/watch?v=MMn2e1iIdck">Video</a></span>
<span class="pub-venue">Social NLP Workshop, ACL 2018</span>
</li>
</ul>

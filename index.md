<img class="home-photo" src="{{ site.baseurl }}/assets/me.jpeg" alt="Puneet Mathur">

I am a Research Scientist at [Adobe Research](https://research.adobe.com/) (AIR Lab) in San Jose, CA. My research spans **agentic AI evaluations, benchmarking, and reinforcement learning environments**, as well as **audio language modeling and full-duplex speech systems**. I completed my Ph.D. in Computer Science at the [University of Maryland, College Park](https://www.cs.umd.edu/), advised by [Dr. Dinesh Manocha](https://www.cs.umd.edu/people/dmanocha), and an M.S. in Computer Science from UMD (2021).

My work on **agentic AI evaluations and reinforcement learning** focuses on building realistic benchmarks, interactive environments, and evaluation frameworks for multimodal agents. I am particularly interested in assessing complex, multi-step workflows involving reasoning, tool use, and decision-making, and developing environments that support reinforcement learning-based post-training and continuous improvement of agent capabilities.

My research in **audio language modeling and full-duplex speech systems** focuses on enabling natural, real-time spoken interactions. I study models that can listen, reason, and speak simultaneously, handle interruptions, backchannels, and conversational turn-taking, and operate beyond conventional turn-based speech pipelines. This builds on my earlier work in speech recognition personalization and neural speech synthesis.

I maintain a full list of my publications on the [Research]({{ site.baseurl }}/research/) page. I'm always happy to hear about potential research collaborations — feel free to reach out!

[Email](mailto:{{ site.email }}) / [CV]({{ site.baseurl }}/assets/CV.pdf) / [Google Scholar]({{ site.scholar_url }}) / [LinkedIn](https://www.linkedin.com/in/{{ site.linkedin_username }}/) / [GitHub](https://github.com/{{ site.github_username }})


#### Updates

<div class="updates-table">
<table>
<colgroup>
  <col style="width: 105px;">
  <col>
</colgroup>

<!-- Latest blog posts: automatically updated -->
{% for post in site.posts limit:5 %}
<tr>
  <td><b>{{ post.date | date: "%b %Y" }}:</b></td>
  <td>New blog post: <a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a>.</td>
</tr>
{% endfor %}

<!-- Latest research: 2026 -->

<tr>
  <td><b>Oct 2026:</b></td>
  <td>New paper: <a href="https://arxiv.org/abs/2610.20172" target="_blank">MultiPartyTurnBench: Multi-Party Turn-Taking and Interruption Detection</a> — in submission. <a href="https://huggingface.co/datasets/puneetUMD/MultiPartyTurnBench" target="_blank">[Dataset]</a></td>
</tr>

<tr>
  <td><b>Sep 2026:</b></td>
  <td>New paper: <a href="https://arxiv.org/abs/2609.34973" target="_blank">APEX-Voice: Can Voice Agents Complete Professional Workflows Through Full-Duplex Interaction</a> — in submission. <a href="https://apex-voice.github.io/" target="_blank">[Project Page]</a></td>
</tr>

<tr>
  <td><b>Sep 2026:</b></td>
  <td>New paper: <a href="https://arxiv.org/abs/2609.35791" target="_blank">FD-VAD: Semantic Endpoint Detection for Streaming Full-Duplex Speech</a> — in submission. <a href="https://fd-vad.github.io/" target="_blank">[Project Page]</a></td>
</tr>

<tr>
  <td><b>2026:</b></td>
  <td>New paper: <b>Duplex-R1: Full-Duplex Audio LMs that Reason, Retrieve, and Speak While Searching</b> — in submission. [Coming soon]</td>
</tr>

<tr>
  <td><b>2026:</b></td>
  <td>New paper: <a href="https://arxiv.org/pdf/2609.03423" target="_blank">DuplexSpeechBench–IFEval: Evaluating Implicit Instruction Following in Full-Duplex Voice Agents</a> — in submission. <a href="https://dsb-ifeval.github.io/" target="_blank">[Project Page]</a></td>
</tr>

<tr>
  <td><b>2026:</b></td>
  <td>New paper: <a href="https://arxiv.org/pdf/2610.00316" target="_blank">DuplexSpeechBench–Document Grounding: Benchmarking Document Grounding and Hallucinations in Full-Duplex Voice Agents</a> — in submission. <a href="https://dsb-dg.github.io/" target="_blank">[Project Page]</a></td>
</tr>

<tr>
  <td><b>Aug 2026:</b></td>
  <td>New paper: <a href="https://arxiv.org/pdf/2608.09227" target="_blank">Omni2LoRA: Coherence-Preserving Parametric Memory for Efficient Omni Language Models</a> — in submission. <a href="https://omni2lora.github.io/" target="_blank">[Project Page]</a></td>
</tr>

<tr>
  <td><b>2026:</b></td>
  <td>New paper: <a href="https://arxiv.org/abs/2604.25231" target="_blank">DRAGON: A Benchmark for Evidence-Grounded Visual Reasoning over Diagrams</a> — NeurIPS 2026.</td>
</tr>

<tr>
  <td><b>Mar 2026:</b></td>
  <td>New paper: <a href="https://arxiv.org/abs/2603.06138" target="_blank">Partial Policy Gradients for RL in LLMs</a> — in submission.</td>
</tr>

<tr>
  <td><b>Mar 2026:</b></td>
  <td>New paper: <a href="https://arxiv.org/abs/2603.23518" target="_blank">Cluster-R1: Large Reasoning Models Are Instruction-following Clustering Agents</a> — in submission.</td>
</tr>

<tr>
  <td><b>2026:</b></td>
  <td><a href="https://arxiv.org/abs/2511.08798" target="_blank">ARGUS: Structured Uncertainty Guided Clarification for LLM Agents</a> accepted to ACL 2026.</td>
</tr>

</table>
</div>

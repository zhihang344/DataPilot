# Project 2
Every student should submit the code, report (in pdf format) and ppt on Blackboard
system before 23:59 June 2. Report can be written either in English or Chinese. Name
your report as studentID_Name.pdf and your code as studentID_name.zip .

## Q1. Data Agent System
Design and build a generative Data Agent system. A user without expert knowledge in
programming or data science can directly obtain end-to-end data analysis pipelines,
mathematical problem abstraction, and predictive models by prompting the LLM with
natural questions. The training dataset is from [DataMind-12K](https://huggingface.co/datasets/zjunlp/DataMind-12K), a high-quality trajectory
set specifically designed for Supervised Fine-Tuning (SFT) of data-analytic agents.

(1) [3 points] Read [A Survey on Data Selection for Language Models](https://github.com/alon-albalak/data-selection-survey?tab=readme-ov-file#data-selection-for-instruction-tuning-and-multitask-training) (Data Selection for
Instruction-Tuning and Multitask Training Part). Illustrate three methods in detail that
you consider most effective for data selection within the data science and mathematical
modeling domain.

(2) [3 points] Write python code to process data. Download the datamind_12k.json file
from the DataMind-12K repository. Select 2k samples from this JSON file as training
data, and 500 samples as validation data. Use the agent trajectory selection/filtering
methods you illustrated in task (1) (such as complexity-based filtering, trajectory
deduplication, or reward-based selection, etc.) rather than just randomly sampling.
Prepare the selected data according to the requirements of [Qwen model training](https://github.com/QwenLM/Qwen3). You
may want to use free APIs from [GLM-4.7-Flash](https://bigmodel.cn/pricing).

(3) [3 points] Use [Ray-train python code](https://docs.ray.io/en/latest/train/train.html) or [Open-RLHF toolkit](https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_sft.sh) to train the Data Agent
model by further finetuning the Qwen3.5-0.8B model. As no GPU is available in the
server, you can use pytorch-cpu to debug your code, train the model for a few hours,
and save a checkpoint. Note that the validation set is used for designing the
hyperparameters and selecting the model checkpoint. You can also rent the GPU server
in the [AutoDL](https://www.autodl.com/login) platform or use GPUs from google colab.

(4) [1 points] Deploy the agent model with your laptop and prepare a website for the
demo. You can use the official Qwen3.5-0.8B checkpoint in this demo. You can refer to
the tutorials [here](https://github.com/QwenLM/Qwen3/blob/main/examples/demo/web_demo.py). Note that you may need to change the web UI to better support data
analysis task.

## Q2. Startup Business Plan
The advancement of large language models makes it possible that small teams can
achieve significant business success in various startups, such as Midjourney and Meshy.

(1) [3 points] Suppose you plan to build a profitable startup focusing on LLM
applications in the indie way (see reference [here](https://sideproject.guide/)). Brainstorm the idea for a startup and
write a business plan (see references [here](https://www.jbs.cam.ac.uk/wp-content/uploads/2020/08/how-to-write-a-business-plan.pdf) and [here](https://zhuanlan.zhihu.com/p/21845926)) with the help of LLM. Survey and
compare your startup with the main competitors in the market. Also write a [roadshow presentation ppt](https://zhuanlan.zhihu.com/p/24545659) to secure funding.

You can refer to the ideas on Product Hunt as well as the suggestions on how to startup
a company.

Please claim the LLM system you use in your report. You are responsible to check the
content and make sure that the content is correct. Write all the questions/prompts you
use to chat with the bot.

(2) [3 points] Prompt LLMs to design the system architecture of your product, to
support industrial-grade deployment and 100,000-level concurrency. The system has
modules like LLM engine, data processing, database, modules to support high
concurrency, monitoring and operation Module, etc. Write an architecture design
document with the system design diagram.

You can search references like [AI-System-Design](https://www.systemdesignhandbook.com/guides/ai-system-design/) and [System-Design-Primer](https://github.com/donnemartin/system-design-primer) and
[System-Design-101](https://github.com/ByteByteGoHq/system-design-101).

## Presentation
[4 points] Presentations are on June 3 & June 4. Each sampled student has 12 minutes.
1~2 related questions will be asked after the presentation. Those who have not
presented for Project 1 will have higher priority this time.
"""
LangChain chains for the Module Classification System.
"""
from typing import Dict, Any, List
import logging

from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_nvidia import register_model, Model

from config import config
from utils import initialize_clients, load_vector_store, setup_retrievers

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_classification_chain():
    """
    Create a LangChain chain for module classification.
    
    Returns:
        A runnable chain that performs classification
    """
    # Initialize clients and retrievers
    chat_client, embed_client, parent_splitter, child_splitter, store = initialize_clients()
    vector_store = load_vector_store(embed_client)
    
    register_model(Model(
        id="nvdev/nvidia/llama-3.2-nv-rerankqa-1b-v2",
        model_type="ranking",
        client="NVIDIARerank",
        endpoint="https://ai.api.nvidia.com/v1/nvdev/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
    )
)
    # Initialize reranker with fallback
    try:
        reranker = NVIDIARerank(
            model=config.rerank_model,
            api_key=config.nvidia_api_key,
            top_n=config.top_n_rerank
        )
        logger.info("NVIDIA Reranker initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize NVIDIA Reranker: {str(e)}. Proceeding without reranking.")
        reranker = None
    
    # Set up retrievers
    active_retriever = setup_retrievers(
        vector_store, store, parent_splitter, child_splitter, chat_client, embed_client
    )
    
    # Create the few-shot examples
    examples = [
        {
            "question": r"Synopsis : [SDLC4.0 EG SOL KPI][MMAutomation] edge browser Netflix login not working, Description: Netflix automation from PlayReadyDRM class, login not working.",
            "answer": r"""
Classification: script
Confidence: High
Key Supporting Points:
• The query mentions "MMAutomation", which suggests automation scripting.
• It specifically talks about "Netflix automation from PlayReadyDRM class", indicating a scripted approach to automate a process.
• The context revolves around resolving an issue (login not working), which is a common focus of scripting in automation scenarios.
"""
        },
        {
            "question": r"""Synopsis : [R565][Security EG][Display Scaling]: Drop down is grey out and unable to select display scaling.	System Details:  DT AD106  WIndows 24H2  566.24-sandbag_security.  Description: Repro steps:  1.Install driver.  2.Enable DSR resolution and set Image scaling to On.  3.Go to adjust desktop size and scaling page in nvcpl -&gt; try to set display scaling    4.Observe.    Observation:  Scaling dropdown is grey out.    Repro Rate : 2/2    Attachments    diagnostics.txt  dispdiag  wpp logs""",
            "answer": r"""
Classification: product
Confidence: High
Key Supporting Points:
• The query discusses a specific issue with a product (driver) related to display scaling.
• It mentions system details, including a specific device model (DT AD106) and operating system (Windows 24H2), which is typical for product-related issues.
• The query includes repro steps and observations, which is consistent with product testing and troubleshooting.
• Attachments such as diagnostics.txt, dispdiag, and wpp logs are mentioned, which are typical for product issue investigations.
"""
        },
        {
            "question": r"""Synopsis: [Atp-Infrastructure] Driver copy failed for machine ATP-MM-DL-HDCP-04, Description: ISSUE :  <table border="1" cellspacing="0" id="gridLinkedTasks" rules="all">  <tbody>  <tr>   <td>ERROR: Copying OR Unzipping driver failed. Please verify driver existed at the time of test on following path: \\builds.nvidia.com\Nightly\NV\wddm2-x64-dch\r570_00-241104-35077054-sandbag\UDA\GeforceWeb\Public\International</td>  </tr>  </tbody>  </table>  file://pu-cdot02-corp01/atpdatapune/ATP/Results/TASK/37996/379967360  MAchine NAme : ATP-MM-DL-HDCP-04""",
            "answer": r"""
Classification: script
Confidence: High
Key Supporting Points:
• The query contains a specific error message related to a technical process ("Driver copy failed for machine ATP-MM-DL-HDCP-04"), indicating it is related to a technical or automated process.
• The mention of a file path ("\builds.nvidia.com\Nightly\NV\wddm2-x64-dch\r570_00-241104-35077054-sandbag\UDA\GeforceWeb\Public\International") suggests involvement of a script or automated tool.
• The structured format of the error message, including a table with a specific error description ("ERROR: Copying OR Unzipping driver failed"), further supports that it is generated by a script.
• The inclusion of a machine name ("ATP-MM-DL-HDCP-04") and a file link ("file://pu-cdot02-corp01/atpdatapune/ATP/Results/TASK/37996/379967360") points towards an automated or scripted process for managing or reporting technical tasks or errors.
"""
        },
        {
            "question": r"""[Lenovo GA106][GPUMon]:Unable to launch GPUmon ,fails with error "Failed to set the PCI-E interface "	**System Info**    </strong> <span style="background-color:rgb(255,255,255);color:rgb(5,5,5);">OS: Windows  Win 11 SV2</span></span> <span style="background-color:rgb(255,255,255);color:rgb(5,5,5);"> RAM: 32 GB</span></span> <span style="background-color:rgb(255,255,255);color:rgb(5,5,5);"> GPU: G106</span></span> <span style="background-color:rgb(255,255,255);color:rgb(5,5,5);"> Display: LVDS</span></span> <span style="background-color:rgb(255,255,255);color:rgb(5,5,5);"> Driver: 551.14 path : \\builds.nvidia.com\Prerelease\AttestedDriverSigning\Attested_logod\wddm2-x64-dch\551.14-sandbag\UDA\GeforceWeb\Public\International</span></span>  **Repro steps**</strong><span style="background-color:rgb(255,255,255);color:rgb(5,5,5);">    </span></span>  1. Install driver under test.  2. Copy latest GPUMon . 3. Try to Launch the GPUMon  4.Observed  *Observation**     </strong> <span style="background-color:rgb(255,255,255);color:rgb(5,5,5);">1)Unable to launch it . giving error : ""Failed to set the PCI-E interface &quot;</span></span> **Expected*</strong> Should able to Launch it   **ISolation*</strong>   1. Issue no repro with Dell AD104  2. Issue no repro with Alienware GA106  2. Issue no repro with Alienware GA104 **regression**   </strong> <span style="background-color:rgb(255,255,255);color:rgb(5,5,5);">issue repro with546.29, 546.52, 551.08, 545.94</span></span> <span style="background-color:rgb(255,255,255);color:rgb(5,5,5);">Marking as not  a Regression.</span></span>  **Repro rate**   </strong><span style="background-color:rgb(255,255,255);color:rgb(5,5,5);">             </span></span> <span style="background-color:rgb(255,255,255);color:rgb(5,5,5);">2/2(100%)   </span></span>  **Attachments**       </strong> <span style="background-color:rgb(255,255,255);color:rgb(5,5,5);">1.Repro Image</span></span>""",
            "answer": r"""
Classification: product
Confidence: High
Key Supporting Points:
• The query is related to a specific hardware issue with a Lenovo GA106 GPU and a driver version, which suggests a product-related problem.
• The system information provided includes details about the OS, RAM, GPU, display, and driver, which is typical for product-related issues.
• The repro steps and observations are focused on the functionality of the GPUMon tool, which is a product-specific feature.
• The isolation and regression sections discuss the issue in relation to specific hardware and driver versions, further supporting the classification as a product-related issue.
"""
        },
        {
            "question": r"""[Colossus Platform] Reprovision Lease request for ipp2-0458(IPP-2-U1) keeps on Failing in the Resource Configuration	<h1> </h1>  The Reprovision Lease request for the machine &quot;<strong>ipp2-0458(IPP-2-U1)</strong>&quot; keeps on failing in the resource configuration. Could you please look into the issue. Please find below the details  Reprovision API -   <strong>https://colossus.nvidia.com/v3/lease/31cdd02e-1039-49e8-aa2a-6357dc08f82d/reprovision</strong>    <strong>Lease ID</strong> - 31cdd02e-1039-49e8-aa2a-6357dc08f82d    <strong>Request ID</strong> - 74818ed0-c304-4393-845b-16b221de8317      Output from the API <strong>https://colossus.nvidia.com/v3/request/74818ed0-c304-4393-845b-16b221de8317/configjobs</strong>    [      {          &quot;jobId&quot;: 243887,          &quot;awxServerId&quot;: &quot;awx-portal3&quot;,          &quot;playbookName&quot;: &quot;hardreleaseplaybook.yml&quot;,          &quot;status&quot;: &quot;SUCCESSFUL&quot;,          &quot;startTime&quot;: &quot;2023-07-06T03:07:41.000+00:00&quot;,          &quot;endTime&quot;: &quot;2023-07-06T03:08:09.000+00:00&quot;,          &quot;logsUrl&quot;: &quot;v3/request/74818ed0-c304-4393-845b-16b221de8317/configjobs/awx-portal3/243887/log&quot;      },      {          &quot;jobId&quot;: 243921,          &quot;awxServerId&quot;: &quot;awx-portal3&quot;,          &quot;playbookName&quot;: &quot;winbaseplaybook.yml&quot;,          &quot;status&quot;: &quot;SUCCESSFUL&quot;,          &quot;startTime&quot;: &quot;2023-07-06T03:32:55.000+00:00&quot;,          &quot;endTime&quot;: &quot;2023-07-06T03:37:55.000+00:00&quot;,          &quot;logsUrl&quot;: &quot;v3/request/74818ed0-c304-4393-845b-16b221de8317/configjobs/awx-portal3/243921/log&quot;      },      {          &quot;jobId&quot;: 243931,          &quot;awxServerId&quot;: &quot;awx-portal3&quot;,          &quot;playbookName&quot;: &quot;atp-install.yml&quot;,          &quot;status&quot;: &quot;FAILED&quot;,          &quot;startTime&quot;: &quot;2023-07-06T03:38:21.000+00:00&quot;,          &quot;endTime&quot;: null,          &quot;logsUrl&quot;: &quot;v3/request/74818ed0-c304-4393-845b-16b221de8317/configjobs/awx-portal3/243931/log&quot;      },      {          &quot;jobId&quot;: 339879,          &quot;awxServerId&quot;: &quot;awx-portal2&quot;,          &quot;playbookName&quot;: &quot;local-account-setup.yml&quot;,          &quot;status&quot;: &quot;SUCCESSFUL&quot;,          &quot;startTime&quot;: &quot;2023-07-06T03:30:06.000+00:00&quot;,          &quot;endTime&quot;: &quot;2023-07-06T03:32:50.000+00:00&quot;,          &quot;logsUrl&quot;: &quot;v3/request/74818ed0-c304-4393-845b-16b221de8317/configjobs/awx-portal2/339879/log&quot;      },      {          &quot;jobId&quot;: 339902,          &quot;awxServerId&quot;: &quot;awx-portal2&quot;,          &quot;playbookName&quot;: &quot;vbios-restore-playbook.yml&quot;,          &quot;status&quot;: &quot;FAILED&quot;,          &quot;startTime&quot;: &quot;2023-07-06T03:37:59.000+00:00&quot;,          &quot;endTime&quot;: null,          &quot;logsUrl&quot;: &quot;v3/request/74818ed0-c304-4393-845b-16b221de8317/configjobs/awx-portal2/339902/log&quot;      }  ]""",
            "answer": r"""
Classification: product
Confidence: High
Key Supporting Points:
• The query is related to a specific product/platform, namely "Colossus Platform".
• The issue is with a reprovision lease request for a machine, which suggests a product-specific problem.
• The query includes technical details such as API URLs, lease IDs, and request IDs, which are typical of product-related issues.
• The output from the API includes job IDs, playbook names, and statuses, which further supports the classification as a product issue.
"""
        }
    ]
    
    # Create a standalone prompt template for few-shot prompting
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "Question: {question}"),
        ("ai", "{answer}")
    ])
    
    # Use FewShotChatMessagePromptTemplate
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )
    
    # Create the final prompt by combining system message and few-shot examples
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Based on the following reasoning and observations, classify the module type:
        
        Provide your final classification in this format:
        - Classification: [script/product](must be in lower case always)
        - Confidence: [High/Medium/Low]
        - Key Supporting Points: (bullet points)
        """),
        few_shot_prompt,
        ("human", "{question}")
    ])
    
    # Create a processing chain to retrieve and rerank documents
    processing_chain = (
        RunnablePassthrough.assign(
            original_question=lambda x: x["question"]
        )
        .assign(
            context=lambda x: active_retriever.invoke(x["original_question"])
        )
        .assign(
            reranked_context=lambda x: reranker.compress_documents(
                query=x["original_question"],
                documents=x["context"]
            ) if reranker else x["context"]
        )
    )
    
    # Create the classification chain
    classification_chain = (
        final_prompt 
        | chat_client 
        | StrOutputParser()
    )
    
    # Create the full chain that combines processing and classification
    full_chain = (
        RunnablePassthrough.assign(
            processed=lambda x: processing_chain.invoke({"question": x["question"]})
        )
        .assign(
            classification=lambda x: classification_chain.invoke({"question": x["question"]})
        )
    )
    
    return full_chain

def parse_classification_result(result: str) -> Dict[str, Any]:
    """
    Parse the classification result string into structured data.
    
    Args:
        result: The raw classification result string
        
    Returns:
        Dict with classification, confidence, and supporting points
    """
    lines = result.strip().split('\n')
    classification = ""
    confidence = ""
    supporting_points = []
    
    for line in lines:
        line = line.strip()
        if line.startswith("Classification:"):
            classification = line.split("Classification:")[1].strip()
        elif line.startswith("Confidence:"):
            confidence = line.split("Confidence:")[1].strip()
        elif line.startswith("•") or line.startswith("*"):
            supporting_points.append(line.replace("•", "").replace("*", "").strip())
    
    return {
        "classification": classification,
        "confidence": confidence,
        "supporting_points": supporting_points
    } 
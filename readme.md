# VLA/VLM/LLM/Diffusion + RL for End-to-End Autnonomous Driving(E2EAD)
## Benchmark
### Open-loop

+ [![Github](https://img.shields.io/badge/Github-NavSimV2-yellow?&logo=github&labelColor=305ce5)](https://github.com/autonomousvision/navsim)<details><summary>NavSim</summary>Based on nuPlan dataset, employs Predictive Driver Model Score (PDMS) to assess key aspects of driving behavior, such as collision and ego progress.</details>

<!-- + NavSim: [![Github](https://img.shields.io/badge/Github-NavSimV2-yellow?&logo=github&labelColor=305ce5)](https://github.com/autonomousvision/navsim)
  + based on nuPlan dataset, employs Predictive Driver Model Score (PDMS) to assess key aspects of driving behavior, such as collision and ego progress. -->
+ [![Website](https://img.shields.io/badge/Project-NuScenes-blue)](https://www.nuscenes.org/)<details><summary>nuScenes</summary>It uses L2 distance and collision rate as evaluation metrics.</details>
+ [![Website](https://img.shields.io/badge/Project-Waymo-blue)](https://waymo.com/open/data/e2e/)<details><summary>Waymo E2E driving benchmark</summary> It uses Rater Feedback Score (RFS), which reflects human-judged planning quality.</details>


### Closed-loop
+ <details><summary>Bench2Drive</summary>

  working within CARLA simulator. Evaluate on planning and reasonging ability. It contains 44 interactive, closed-loop scenarios under varying locations and weather conditions, using metrics such as 
    + success rate(SR): percentage of routes completed successfully within the allocated time and without committing any traffic violations.
    + driving score(DS): the product of Route Completion and Infraction Score, capturing both task completion and rule adherence
    + efficiency: quantifies the vehicle’s velocity relative to surrounding traffic, encouraging progressiveness without aggression. 
    + comfort: reflects the smoothness of the driving trajectory
    + Multi-Ability: Merging, Overtakeing, Emergency Braking, Yielding and Traffic Signs
  </details>
+ <details><summary>nuPlan closed-loop planning</summary>
  
  + overall planning score; 
  + collision score; 
  + progress score </details>
  
## Dataset
+ <details><summary>nuPlan</summary>contains 120 hours of large-scale driving data with eight streams of camera data and object annotations.</details>
+ <details><summary>nuScences</summary>1,000 urban driving scenes with six camera views. contains 1000 scenes from Singapore and Boston, with 700 scenes for training, 150 scenes for validation, and 150 scenes for testing. Each scene spans 20 seconds and is annotated at 2 Hz.</details>
+ <details><summary>Waymo E2E</summary>4,021 20-second driving segments with eight streams of camera views and ego vehicle trajectories, especially focusing on challenging and **long-tail scenarios**, such as driving through construction areas or risky situations.</details>
+ <details><summary>DriveLM</summary>a VQA dataset built on nuScnenes and CARLA simulation data.</details>
+ <details><summary>CARLA-Garage</summary>provides over 500,000 frames of camera data from the CARLA simulator.</details>
+ <details><summary>Bench2Drive</summary>

    + includes 220 short routes for evaluation, with one challenging case per route for different driving abilities
    + Training set (base) includes a total of 1000 clips, 950 for training, 50 for validation.
  </details>
+ <details><summary>OpenScene</summary>a compact subset of the nuPlan dataset sampled at 2Hz.</details>
## Paper

### VLA+RL
1. [2025-06-16] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning [![arXiv](https://img.shields.io/badge/arXiv-AutoVLA-black?logoColor=white&labelColor=8f1616&logo=arXiv&style=plastic)](https://arxiv.org/abs/2506.13757) [![Website](https://img.shields.io/badge/Project-AutoVLA-blue)](https://autovla.github.io/)
   + A unified autoregressive VLA model with dual-thinking(fast/slow with CoT) response.
   + <details><summary>More Info</summary>

     + Additional work: action codebook containing 2048 discrete action tokens; Curate a large causal reasoning annotations(CoT data)
     + Input: 
       + Text(Navigation instructions); 
       + Ego states(velocity/Acceleration/History Action); 
       + Multi-view Image Streams(2Hz)
     + Output(1Hz): 
       + reasoning texts and action tokens; 
       + also tested to predict text waypoints, underperforming physical action prediction.
     + Backbone: Qwen2.5-VL-3B
     + Training: fine-tuned on a mixture of CoT driving data and sole action scenarios with **a combination loss of prediction** on text token and action token; 
     + RL post training: reward function $r=R_{score}-\lambda \cdot r_{CoT}$ based on benchmark score, $r_{CoT}$  penalizes the length of CoT, trained with **GRPO** to get the adaptive ability switching between fast and slow response
     </details>
2. [2025-05-22] DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving [![arXiv](https://img.shields.io/badge/arXiv-DriveMoE-black?logoColor=white&labelColor=8f1616&logo=arXiv&style=plastic)](https://arxiv.org/abs/2505.16278) [![Website](https://img.shields.io/badge/Project-DriveMoE-blue)](https://thinklab-sjtu.github.io/DriveMoE/) [![Github](https://img.shields.io/badge/Github-❌-lightgrey?&logo=github&labelColor=305ce5)](https://github.com/Thinklab-SJTU/DriveMoE)
   + Add vision and action MoE modules into Drive $\pi_0$ to obtain DriveMoE
     + vision MoE: fiexed views + selective views based on the current driving context
     + action MoE: replace FFNs with MoE layers within flow-matching transformer.
   + <details><summary>More Info</summary>

      + Training Drive $\pi_{0}$
        + Input: 2 fixed sequential front-view images; vehicle states(position, velocity, acceleration and heading angle)
        + Output: 10 future waypoints
        + fine-tuning standard $\pi_{0}$ on training set of Bench2Drive.
      + Training DriveMoE:
        + Input: 
          + Textual prompt;
          + vehicle state(speed, yaw rate, past trajectory)
          + A sequence of surround-view iamges asked by vision MoE(2 frames of fixed view + 1 optional frame from another view)
        + Output:
          + Action MoE utilized top-3 of (1+6) erperts to generate 10 future waypoints
        + Stage 1: train all routers(provide proper experts) and experts(generate proper action) of both MoE supervised by ground-truth experts.
        + Stage 2: continued to training of stage 1, but removing ground-truth experts
      + Backbone: Paligemma VLM 3b-pt-224
      + Additional work:
        + Annotate camera-level selection instruction to supervise for vision MoE</details>
### VLM+RL
3. [2025-03-25] ![Conference](https://img.shields.io/badge/ICCV2025-conference?color=red)ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation [![arXiv](https://img.shields.io/badge/arXiv-ORION-black?logoColor=white&labelColor=8f1616&logo=arXiv&style=plastic)](https://arxiv.org/abs/2503.19755) [![Website](https://img.shields.io/badge/Project-ORION-blue)](https://xiaomi-mlab.github.io/Orion/) [![Github](https://img.shields.io/badge/Github-ORION-yellow?&logo=github&labelColor=305ce5)](https://github.com/xiaomi-mlab/Orion)
    + A VAE/Diffusion-based generative planner conditioning on reasoning space of LLM  and action space of trajectory
    + <details><summary>More Info</summary>
    
      + Following Query-Former of OmniDrive to use learnable queries including scene, perception, and history.
      + Training:
         1. align vision with reasoning: Train QT-Former and VLM while freezing generative model on VQA data of Chat-B2D
         2. Transfer world knowledge to action space: Train the whole model except LLM on planning tasks
         3. Jointly training on VQA and planning
      + Additional data: Use Qwen2-VL to annotate VQA from Bench2Drive dataset, resulting in Chat-B2D to fine-tune LoRA
      + Backbone: EVA-02-L as vision encoder; Vicuna v1.5 as LLM
      </details>
### Diffusion + RL
4. [2025-03-13] Finetuning Generative Trajectory Model with Reinforcement Learning from Human Feedback [![arXiv](https://img.shields.io/badge/arXiv-TrajHF-black?logoColor=white&labelColor=8f1616&logo=arXiv&style=plastic)](https://arxiv.org/abs/2503.10434)
   + Diffusion transformer post-trained by GRPO on Human Preference dataset, behavioral cloning pretrained model on general dataset.
   + <details><summary>More Info</summary>
   
     + Dataset: NavSim/LiAuto Normal dataset for pretraining; LiAuto Preference dataset for HFRL
     + Input: front-view images; LiDAR sensors, historical actions, and ego states
     + Output: 8 waypoints spanning 4 seconds
     + ViT as iamge encoder; ResNet34 as LiDAR encoder;
     </details>
5. [2024-10-08] ![Conference](https://img.shields.io/badge/ICRA2025-conference?color=red) Gen-Drive: Enhancing Diffusion Generative Driving Policies with Reward Modeling and Reinforcement Learning Fine-tuning [![arXiv](https://img.shields.io/badge/arXiv-GenDrive-black?logoColor=white&labelColor=8f1616&logo=arXiv&style=plastic)](https://arxiv.org/abs/2410.05582) [![Website](https://img.shields.io/badge/Project-GenDrive-blue)](https://mczhi.github.io/GenDrive/)
    + Generative planner (Query + Diffusion transformer) fine-tuned by a reward model (trained on preference data) and RL with Denosing Diffusion PO
    + <details><summary>More Info</summary>

      + Input: 
        + trajectory of surroundings; 
        + map polylines of surroundings;
        + 30 polylines of ego vehicle;
      + Output: 
        + future trajectories of ego vehicle and objects closest to ego vehicle
      + Additional work:
        + Train reward model
        + Curate a pairwise preference dataset using GPT-4o for reward model training
      </details>

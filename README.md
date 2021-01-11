# Awesome-AutoML
Curating a list of AutoML-related research, tools, projects and other resources

# AutoML

AutoML is the tools and technology to use machine learning methods and processes to automate machine learning systems and make them more accessible. It existed for several decades so it's not a completely new idea. 

Recent work by Google Brain and many others have re-kindled the enthusiasm of AutoML and some companies have already [commercialized the technology](https://cloud.google.com/automl/). Thus, it has becomes one of the hosttest areas to look into. 

There are many kinds of AutoML, including:
- Neural network architecture search
- Hyperparameter optimization
- Optimizer search
- Data augmentation search
- Learning to learn/Meta-learning
- And many more

## Research

### AutoML survey
- [A Comprehensive Survey of Neural Architecture Search: Challenges and Solutions](https://arxiv.org/pdf/2006.02903.pdf) (Ren et al. 2020)
- [AutoML: A Survey of the State-of-the-Art](https://arxiv.org/abs/1908.00709) (He et al. 2019)
- [A Survey on Neural Architecture Search](https://arxiv.org/abs/1905.01392) (Wistuba et al. 2019)
- [Neural Architecture Search: A Survey](https://arxiv.org/abs/1808.05377) (Elsken et al. 2019)
- [Taking Human out of Learning Applications: A Survey on Automated Machine Learning](https://arxiv.org/abs/1810.13306) (Yao et al. 2018)

### Neural Architecture Search
- [CurveLane-NAS: Unifying Lane-Sensitive Architecture Search and Adaptive Point Blending](https://arxiv.org/pdf/2007.12147.pdf) (Xu et al. 2020)
- [Cream of the Crop: Distilling Prioritized Paths For One-Shot Neural Architecture Search](https://arxiv.org/abs/2010.15821) (Peng et al. 2020)
- [CurveLane-NAS: Unifying Lane-Sensitive Architecture Search and Adaptive Point Blending](https://arxiv.org/abs/2007.12147) (Xu et al. 2020)
- [MCUNet: Tiny Deep Learning on IoT Devices](https://arxiv.org/pdf/2007.10319.pdf) (Lin et al. 2020)
- [FBNetV2: Differentiable Neural Architecture Search for Spatial and Channel Dimensions](https://arxiv.org/abs/2004.05565) (Wan et al. 2020)
- [Neural Architecture Transfer](https://arxiv.org/abs/2005.05859v1) (Lu et al. 2020)
- [When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks](https://arxiv.org/abs/1911.10695) (Guo et al. 2020)
- [Semi-Supervised Neural Architecture Search](https://arxiv.org/abs/2002.10389) (Luo et al. 2020)
- [MixPath: A Unified Approach for One-shot Neural Architecture Search](https://arxiv.org/abs/2001.05887) (Chu et al. 2020)
- [AutoML-Zero: Evolving Machine Learning Algorithms From Scratch](https://arxiv.org/abs/2003.03384) (Real et al. 2020)
- [Generative Teaching Networks: Accelerating Neural Architecture Search by Learning to Generate Synthetic Training Data](https://arxiv.org/abs/1912.07768) (Such et al. 2019)
- [CARS: Continuous Evolution for Efficient Neural Architecture Search](https://arxiv.org/abs/1909.04977) (Yang et al. 2019)
- [Meta-Learning of Neural Architectures for Few-Shot Learning](https://arxiv.org/abs/1911.11090) (Elsken et al. 2019)
- [Up to two billion times acceleration of scientific simulations with deep neural architecture search](https://arxiv.org/abs/2001.08055) (Kasim et al. 2019)
- [Efficient Forward Architecture Search](https://www.microsoft.com/en-us/research/publication/efficient-forward-architecture-search/) (Hue et al. 2019)
- [NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection](https://arxiv.org/abs/1904.07392) (Ghiasi et al. 2019)
- [Improving Keyword Spotting and Language Identification via Neural Architecture Search at Scale](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/1916.html) (Mazzawi et al. 2019)
- [SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization](https://arxiv.org/abs/1912.05027) (Du et al. 2019)
- [Evaluating the Search Phase of Neural Architecture Search](https://arxiv.org/abs/1902.08142) (Sciuto et al. 2019)
- [MixConv: Mixed Depthwise Convolutional Kernels](https://arxiv.org/abs/1907.09595) (Tan et al. 2019)
- [Multinomial Distribution Learning for Effective Neural Architecture Search](https://arxiv.org/abs/1905.07529) (Zheng et al. 2019)
- [SNR: Sub-Network Routing for Flexible Parameter Sharing in Multi-task Learning](https://research.google/pubs/pub47842/) (Ma et al. 2019)
- [PC-DARTS: Partial Channel Connections for Memory-Efficient Differentiable Architecture Search](https://arxiv.org/abs/1907.05737) (Xu et al. 2019) - [code](https://github.com/yuhuixu1993/PC-DARTS)
- [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420) (Guo et al. 2019)
- [AutoGAN: Neural Architecture Search for Generative Adversarial Networks](https://arxiv.org/abs/1908.03835) (Gong et al. 2019)
- [MixConv: Mixed Depthwise Convolutional Kernels](https://arxiv.org/abs/1907.09595?context=cs.LG) (Tan et al. 2019)
- [Tiny Video Networks](https://arxiv.org/abs/1910.06961) (Piergiovanni et al. 2019)
- [AssembleNet: Searching for Multi-Stream Neural Connectivity in Video Architectures](https://arxiv.org/abs/1905.13209) (Ryoo et al. 2019)
- [EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML](https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html) (Gupta et al. 2019)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (Tan et al. 2019)
- [MoGA: Searching Beyond MobileNetV3](https://arxiv.org/abs/1908.01314) (Chu et al. 2019) - [code](https://github.com/xiaomi-automl/MoGA)
- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) (Howard et al. 2019)
- [Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation](https://arxiv.org/abs/1901.02985) (Liu et al. 2019)
- [DetNAS: Backbone Search for Object Detection](https://arxiv.org/abs/1903.10979) (Chen et al. 2019)
- [Graph HyperNetworks for Neural Architecture Search](https://arxiv.org/pdf/1810.05749.pdf) (Zhang et al. 2019)
- [Dynamic Distribution Pruning for Efficient Network Architecture Search](https://arxiv.org/abs/1905.13543) (Zheng et al. 2019)
- [FairNAS: Rethinking Evaluation Fairness of Weight Sharing Neural Architecture Search](https://arxiv.org/abs/1907.01845) (Chu et al. 2019)
- [SpArSe: Sparse Architecture Search for CNNs on Resource-Constrained Microcontrollers](https://arxiv.org/abs/1905.12107) (Fedorov et al. 2019)
- [EENA: Efficient Evolution of Neural Architecture](https://arxiv.org/abs/1905.07320) (Zhu et al. 2019)
- [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://128.84.21.199/abs/1904.00420) (Guo et al. 2019)
- [InstaNAS: Instance-aware Neural Architecture Search](https://arxiv.org/abs/1811.10201) (Cheng et al. 2019)
- [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/pdf/1812.00332.pdf)  (Cai et al. 2019)
- [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://arxiv.org/abs/1902.09635) (Ying et al. 2019)
- [Evolutionary Neural AutoML for Deep Learning](https://arxiv.org/abs/1902.06827) (Liang et al. 2019)
- [Fast, Accurate and Lightweight Super-Resolution with Neural Architecture Search](https://arxiv.org/abs/1901.07261) (Chu et al. 2019)
- [The Evolved Transformer](https://arxiv.org/abs/1901.11117) (So et al. 2019)
- [SNAS: Stochastic Neural Architecture Search](https://arxiv.org/abs/1812.09926) (Xie et al. 2019)
- [NeuNetS: An Automated Synthesis Engine for Neural Network Design](https://arxiv.org/abs/1901.06261) (Sood et al. 2019)
- [EAT-NAS: Elastic Architecture Transfer for Accelerating Large-scale Neural Architecture Search](https://arxiv.org/abs/1901.05884) (Fang et al. 2019)
- [Understanding and Simplifying One-Shot Architecture Search](https://ai.google/research/pubs/pub47074) (Bender et al. 2018)
- [Evolving Space-Time Neural Architectures for Videos](https://arxiv.org/abs/1811.10636) (Piergiovanni et al. 2018)
- [IRLAS: Inverse Reinforcement Learning for Architecture Search](https://arxiv.org/abs/1812.05285) (Guo et al. 2018)
- [Neural Architecture Search with Bayesian Optimisation and Optimal Transport](https://arxiv.org/abs/1802.07191) (Kandasamy et al. 2018)
- [Path-Level Network Transformation for Efficient Architecture Search](https://arxiv.org/abs/1806.02639) (Cai et al. 2018)
- [BlockQNN: Efficient Block-wise Neural Network Architecture Generation](https://arxiv.org/abs/1808.05584) (Zhong et al. 2018)
- [Stochastic Adaptive Neural Architecture Search for Keyword Spotting](https://arxiv.org/abs/1811.06753v1) (Véniat et al. 2018)
- [Task-Driven Convolutional Recurrent Models of the Visual System](https://arxiv.org/abs/1807.00053) (Nayebi et al. 2018)
- [Neural Architecture Optimization](https://arxiv.org/abs/1808.07233) (Luo et al. 2018)
- [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626) (Tan et al. 2018)
- [Neural Architecture Search: A Survey](https://arxiv.org/abs/1808.05377) (Elsken et al. 2018)
- [MONAS: Multi-Objective Neural Architecture Search using Reinforcement Learning](https://arxiv.org/abs/1806.10332) (Hsu et al. 2018)
- [NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications](https://arxiv.org/abs/1804.03230) (Yang et al. 2018)
- [Auto-Meta: Automated Gradient Based Meta Learner Search](https://arxiv.org/abs/1806.06927) (Kim et al. 2018)
- [MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks](https://arxiv.org/abs/1711.06798) (Gordon et al. 2018)
- [DPP-Net: Device-aware Progressive Search for Pareto-optimal Neural Architectures](https://arxiv.org/abs/1806.08198) (Dong et al. 2018)
- [Searching Toward Pareto-Optimal Device-Aware Neural Architectures](https://arxiv.org/abs/1808.09830) (Cheng et al. 2018)
- [Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) (Liu et al. 2018)
- [Regularized Evolution for Image Classifier Architecture Search](https://arxiv.org/abs/1802.01548) (Real et al. 2018)
- [Efficient Architecture Search by Network Transformation](https://arxiv.org/abs/1707.04873) (Cai et al. 2017)
- [Large-Scale Evolution of Image Classifiers](https://arxiv.org/abs/1703.01041) (Real et al. 2017)
- [Progressive Neural Architecture Search](https://arxiv.org/abs/1712.00559) (Liu et al. 2017)
- [AdaNet: Adaptive Structural Learning of Artificial Neural Networks](https://arxiv.org/abs/1607.01097) (Cortes et al. 2017)
- [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) (Zoph et al. 2017)
- [Designing Neural Network Architectures using Reinforcement Learning](https://arxiv.org/abs/1611.02167) (Baker et al. 2016) 
- [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578) (Zoph and Le. 2016)

### Neural Architecture Search benchmark
- [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://arxiv.org/abs/1902.09635) (Ying et al. 2019) - [code](https://github.com/google-research/nasbench)

### Neural Optimizatizer Search
- [Neural Optimizer Search with Reinforcement Learning](https://arxiv.org/abs/1709.07417) (Bello et al. 2017)

### Activation function Search
- [Searching for Activation Functions](https://arxiv.org/abs/1710.05941) (Ramachandran et al. 2017)

### AutoAugment
- [MetaAugment: Sample-Aware Data Augmentation Policy Learning](https://arxiv.org/abs/2012.12076) (Zhou et al. 2020)
- [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779) (Park et al. 2019)
- [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719) (Cubuk et al. 2019)
- [Learning Data Augmentation Strategies for Object Detection](https://arxiv.org/abs/1906.11172) (Zoph et al. 2019)
- [Fast AutoAugment](https://arxiv.org/abs/1905.00397) (Lim et al. 2019)
- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501) (Cubuk et al. 2018)

### AutoDropout
- [AutoDropout: Learning Dropout Patterns to Regularize Deep Networks](https://arxiv.org/abs/2101.01761) (Pham et al. 2020)


### Learning to learn/Meta-learning
- [ES-MAML: Simple Hessian-Free Meta Learning](https://arxiv.org/abs/1910.01215) (Song et al. 2019)
- [Learning to Learn with Gradients](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2018/EECS-2018-105.html) (Chelsea Finn PhD disseration 2018)
- [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999) (OpenAI Reptile by Nichol et al. 2018)
- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) (MAML by Finn et al. 2017)
- [A sample neural attentive meta-learner](https://arxiv.org/abs/1707.03141) (Mishra et al. 2017)
- [Learning to Learn without Gradient Descent by Gradient Descent](https://arxiv.org/abs/1611.03824) (Chen et al. 2016)
- [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474) (Andrychowicz et al. 2016)
- [Learning to reinforcement learn](https://arxiv.org/abs/1611.05763) (Wang et al. 2016)
- [RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning](https://arxiv.org/abs/1611.02779) (Duan et al. 2016)

### Hyperparameter optimization
- [AutoNE: Hyperparameter Optimization for Massive Network Embedding](https://tadpole.github.io/files/2019_KDD_AutoNE.pdf) (Tu et al. 2019)
- [Population Based Training of Neural Networks](https://arxiv.org/pdf/1711.09846.pdf) (Jaderberg et al. 2017)
- [Google Vizier: A Service for Black-Box Optimization](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf) (Golovin et al. 2017)
- [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560) (Li et al. 2016)
- [Practical Bayesian Optimization of Machine Learning Algorithms](https://arxiv.org/pdf/1206.2944.pdf) (Snoek et al. 2012)
- [Random Search for Hyper-Parameter Optimization](http://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf) (Bergstra et al. 2012)

### Automatic feature selection
- [Deep Feature Synthesis: Towards Automating Data Science Endeavors](https://dai.lids.mit.edu/wp-content/uploads/2017/10/DSAA_DSM_2015.pdf) (Kanter et al. 2017)
- [ExploreKit: Automatic Feature Generation and Selection](https://people.eecs.berkeley.edu/~dawnsong/papers/icdm-2016.pdf) (Katz et al. 2016)

### Model compression
- [AMC: AutoML for Model Compression and Acceleration on Mobile Devices](https://arxiv.org/abs/1802.03494) (He et al. 2018)

## Tools and projects
- [TransmogrifAI](https://github.com/salesforce/TransmogrifAI): an AutoML library written in Scala that runs on top of Apache Spark
- [AutoGluon](https://autogluon.mxnet.io/): AutoML Toolkit for Deep Learning
- [hyperunity](https://github.com/gdikov/hypertunity): A toolset for black-box hyperparameter optimisation
- [auptimizer](https://github.com/LGE-ARC-AdvancedAI/auptimizer): An automatic ML model optimization tool
- [Keras Tuner](https://github.com/keras-team/keras-tuner): Hyperparameter tuning for humans
- [Torchmeta](https://github.com/tristandeleu/pytorch-meta): A Meta-Learning library for PyTorch
- [learn2learn](https://github.com/learnables/learn2learn): PyTorch Meta-learning Framework for Researchers
- [Auto-PyTorch](https://github.com/automl/Auto-PyTorch): Automatic architecture search and hyperparameter optimization for PyTorch
- [ATM: Auto Tune Models](https://hdi-project.github.io/ATM/): A multi-tenant, multi-data system for automated machine learning (model selection and tuning)
- [Adanet: Fast and flexible AutoML with learning guarantees](https://github.com/tensorflow/adanet): Tensorflow package for AdaNet
- [Microsoft Neural Network Intelligence (NNI)](https://github.com/microsoft/nni): An open source AutoML toolkit for neural architecture search and hyper-parameter tuning
- [Dragonfly](https://github.com/dragonfly/dragonfly): An open source python library for scalable Bayesian optimisation
- [H2O AutoML](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html): Automatic Machine Learning by H2O.ai
- [Kubernetes Katib](https://github.com/kubeflow/katib): hyperparameter Tuning on Kubernetes inspired by Google Vizier
- [TransmogrifAI](https://transmogrif.ai/): automated machine learning for structured data by Salesforce
- [Advisor](https://github.com/tobegit3hub/advisor): open-source implementation of Google Vizier for hyper parameters tuning
- [AutoKeras](https://autokeras.com/): AutoML library by Texas A&M University using Bayesian optimization
- [AutoSklearn](https://automl.github.io/auto-sklearn/): an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator
- [Ludwig](https://github.com/uber/ludwig): a toolbox built on top of TensorFlow that allows to train and test deep learning models without the need to write code
- [AutoWeka](http://www.cs.ubc.ca/labs/beta/Projects/autoweka/): hyperparameter search for Weka
- [automl-gs](https://github.com/minimaxir/automl-gs): Provide an input CSV and a target field to predict, generate a model + code to run it
- [SMAC](https://github.com/automl/SMAC3): Sequential Model-based Algorithm Configuration
- [Hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn): hyper-parameter optimization for sklearn
- [Spearmint](https://github.com/HIPS/Spearmint): a software package to perform Bayesian optimization
- [TPOT](http://automl.info/tpot/): one of the very first AutoML methods and open-source software packages 
- [MOE](https://github.com/Yelp/MOE): a global, black box optimization engine for real world metric optimization by Yelp
- [Hyperband](https://github.com/zygmuntz/hyperband): open source code for tuning hyperparams with Hyperband
- [Optuna](https://optuna.org/): define-by-run hypterparameter optimization framework
- [RoBO](https://github.com/automl/RoBO): a Robust Bayesian Optimization framework
- [HpBandSter](https://github.com/automl/HpBandSter): a framework for distributed hyperparameter optimization
- [HPOlib2](https://github.com/automl/HPOlib2): a library for hyperparameter optimization and black box optimization benchmarks
- [Hyperopt](http://hyperopt.github.io/hyperopt/): distributed Asynchronous Hyperparameter Optimization in Python
- [REMBO](https://github.com/ziyuw/rembo): Bayesian optimization in high-dimensions via random embedding
- [ExploreKit](https://people.eecs.berkeley.edu/~dawnsong/papers/icdm-2016.pdf): a framework forautomated feature generation
- [FeatureTools](https://github.com/Featuretools/featuretools): An open source python framework for automated feature engineering
- [PocketFlow](https://github.com/Tencent/PocketFlow): use AutoML to do model compression (open sourced by Tencent)
- [DEvol (DeepEvolution)](https://github.com/joeddav/devol): a basic proof of concept for genetic architecture search in Keras
- [mljar-supervised](https://github.com/mljar/mljar-supervised): AutoML with explanations and markdown reports
- [Determined](https://github.com/determined-ai/determined): scalable deep learning training platform with integrated hyperparameter tuning support; includes Hyperband, PBT, and other search methods
- [AutoGL](https://github.com/THUMNLab/AutoGL): an autoML framework & toolkit for machine learning on graphs)

## Commercial products
- [Amazon SageMaker AutoPilot](https://aws.amazon.com/sagemaker/autopilot/)
- [Google Cloud AutoML](https://cloud.google.com/automl/) 
- [Google Cloud ML Hyperparameter Turning](https://cloud.google.com/ml-engine/docs/tensorflow/using-hyperparameter-tuning)
- [Microsoft Azure Machine Learning Studio](https://azure.microsoft.com/en-us/services/machine-learning-studio/)
- [comet.ml](https://www.comet.ml/)
- [SigOpt](https://sigopt.com/)
- [mljar.com](https://mljar.com)

## Blog posts
- [Neural Architecture Search](https://lilianweng.github.io/lil-log/2020/08/06/neural-architecture-search.html)
- [A Conversation With Quoc Le: The AI Expert Behind Google AutoML](https://medium.com/syncedreview/a-conversation-with-quoc-le-the-ai-expert-behind-google-automl-73a7d0c9fe38)
- [fast.ai: An Opinionated Introduction to AutoML and Neural Architecture Search](https://www.fast.ai/2018/07/12/auto-ml-1/)
- [Introducing AdaNet: Fast and Flexible AutoML with Learning Guarantees](https://ai.googleblog.com/2018/10/introducing-adanet-fast-and-flexible.html)
- [Using Evolutionary AutoML to Discover Neural Network Architectures](https://ai.googleblog.com/2018/03/using-evolutionary-automl-to-discover.html)
- [Improving Deep Learning Performance with AutoAugment](https://ai.googleblog.com/2018/06/improving-deep-learning-performance.html)
- [AutoML for large scale image classification and object detection](https://ai.googleblog.com/2017/11/automl-for-large-scale-image.html)
- [Using Machine Learning to Discover Neural Network Optimizers](https://ai.googleblog.com/2018/03/using-machine-learning-to-discover.html)
- [Using Machine Learning to Explore Neural Network Architecture](https://ai.googleblog.com/2017/05/using-machine-learning-to-explore.html)

## Presentations
- [ICML 2019 Tutorial: Recent Advances in Population-Based Search for Deep Neural Networks](https://www.youtube.com/watch?v=g6HiuEnbwJE) by Evolving AI Lab
- [Automatic Machine Learning](https://videoken.com/embed/5A4xbv5nd8c) by Frank Hutter and Joaquin Vanschoren
- [Advanced Machine Learning Day 3: Neural Architecture Search](https://www.youtube.com/watch?v=wL-p5cjDG64) by Debadeepta Dey (MSR)
- [Neural Architecture Search](https://www.youtube.com/watch?v=sROrvtXnT7Q&t=116s) by Quoc Le (Google Brain)

## Books
- [AUTOML: METHODS, SYSTEMS, CHALLENGES](https://www.automl.org/book/)
- [Hands-On Meta Learning with Python: Meta learning using one-shot learning, MAML, Reptile, and Meta-SGD with TensorFlow](https://www.amazon.com/Hands-Meta-Learning-Python-TensorFlow-ebook/dp/B07KJJHYKF) - [repo](https://github.com/sudharsan13296/Hands-On-Meta-Learning-With-Python)
## Competitions, workshops and conferences
- [NIPS 2018 3rd AutoML Challenge: AutoML for Lifelong Machine Learning](http://automl.chalearn.org/)
- [AutoML Workshop in ICML](https://www.ml4aad.org/workshops/)

## Other curated resources on AutoML
- [Literature on Neural Architecture Search](https://www.ml4aad.org/automl/literature-on-neural-architecture-search/)
- [Awesome-AutoML-Papers](https://github.com/hibayesian/awesome-automl-papers)

# Practical applications
- [AutoML: Automating the design of machine learning models for autonomous driving](https://medium.com/waymo/automl-automating-the-design-of-machine-learning-models-for-autonomous-driving-141a5583ec2a) by Waymo

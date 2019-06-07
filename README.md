# Awesome-AutoML
Curating a list of AutoML-related research, tools, projects and other resources

# AutoML

AutoML is the tools and technology to use machine learning methods and processes to automate machine learning systems and make them more accessible. It existed for several decades so it's not a completely new idea. 

Recent work by Google Brain and many others have re-kindled the enthusiasm of AutoML and some companies have already [commercialized the technology](https://cloud.google.com/automl/). Thus, it has becomes one of the hosttest areas to look into. 

There are many kinds of AutoML. Some applications include:
- Neural network architecture search
- Hyperparameter optimization
- Optimizer search
- Data augmentation search
- Learning to learn/Meta-learning
- And many more

## Research

### AutoML survey
- [Taking Human out of Learning Applications: A Survey on Automated Machine Learning](https://arxiv.org/abs/1810.13306) (Yao et al. 2018)

### Neural Architecture Search
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

### Neural Optimizatizer Search
- [Neural Optimizer Search with Reinforcement Learning](https://arxiv.org/abs/1709.07417) (Bello et al. 2017)

### AutoAugment
- [Fast AutoAugment](https://arxiv.org/abs/1905.00397) (Lim et al. 2019)
- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501) (Cubuk et al. 2018)

### Learning to learn/Meta-learning
- [Learning to Learn with Gradients](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2018/EECS-2018-105.html) (Chelsea Finn PhD disseration 2018)
- [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999) (OpenAI Reptile by Nichol et al. 2018)
- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) (MAML by Finn et al. 2017)
- [A sample neural attentive meta-learner](https://arxiv.org/abs/1707.03141) (Mishra et al. 2017)
- [Learning to Learn without Gradient Descent by Gradient Descent](https://arxiv.org/abs/1611.03824) (Chen et al. 2016)
- [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474) (Andrychowicz et al. 2016)
- [Learning to reinforcement learn](https://arxiv.org/abs/1611.05763) (Wang et al. 2016)
- [RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning](https://arxiv.org/abs/1611.02779) (Duan et al. 2016)

### Hyperparameter optimization
- [Google Vizier: A Service for Black-Box Optimization](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf) (Golovin et al. 2017)
- [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560) (Li et al. 2016)

### Automatic feature selection
- [Deep Feature Synthesis: Towards Automating Data Science Endeavors](https://dai.lids.mit.edu/wp-content/uploads/2017/10/DSAA_DSM_2015.pdf) (Kanter et al. 2017)
- [ExploreKit: Automatic Feature Generation and Selection](https://people.eecs.berkeley.edu/~dawnsong/papers/icdm-2016.pdf) (Katz et al. 2016)

### Model compression
- [AMC: AutoML for Model Compression and Acceleration on Mobile Devices](https://arxiv.org/abs/1802.03494) (He et al. 2018)

## Tools and projects
- [ATM: Auto Tune Models](https://hdi-project.github.io/ATM/): A multi-tenant, multi-data system for automated machine learning (model selection and tuning)
- [Adanet: Fast and flexible AutoML with learning guarantees](https://github.com/tensorflow/adanet): Tensorflow package for AdaNet
- [Microsoft Neural Network Intelligence (NNI)](https://microsoft.github.io/nni/): An open source AutoML toolkit for neural architecture search and hyper-parameter tuning
- [Dragonfly](https://github.com/dragonfly/dragonfly): An open source python library for scalable Bayesian optimisation
- [H2O AutoML](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html): Automatic Machine Learning by H2O.ai
- [Kubernetes Katib](https://github.com/kubeflow/katib): hyperparameter Tuning on Kubernetes inspired by Google Vizier
- [TransmogrifAI](https://transmogrif.ai/): automated machine learning for structured data by Salesforce
- [Advisor](https://github.com/tobegit3hub/advisor): open-source implementation of Google Vizier for hyper parameters tuning
- [AutoKeras](https://autokeras.com/): AutoML library by Texas A&M University using Bayesian optimization
- [AutoSklearn](https://automl.github.io/auto-sklearn/stable/): an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator
- [Ludwig](https://github.com/uber/ludwig): a toolbox built on top of TensorFlow that allows to train and test deep learning models without the need to write code
- [AutoWeka](http://www.cs.ubc.ca/labs/beta/Projects/autoweka/): hyperparameter search for Weka
- [automl-gs](https://github.com/minimaxir/automl-gs): Provide an input CSV and a target field to predict, generate a model + code to run it
- [SMAC](https://github.com/automl/SMAC3): Sequential Model-based Algorithm Configuration
- [Hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn): hyper-parameter optimization for sklearn
- [Spearmint](https://github.com/HIPS/Spearmint): a software package to perform Bayesian optimization
- [TOPT](http://automl.info/tpot/): one of the very first AutoML methods and open-source software packages 
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

## Commercial products
- [Amazon SageMaker](https://aws.amazon.com/sagemaker/)
- [Google Cloud AutoML](https://cloud.google.com/automl/) 
- [Google Cloud ML Hyperparameter Turning](https://cloud.google.com/ml-engine/docs/tensorflow/using-hyperparameter-tuning)
- [Microsoft Azure Machine Learning Studio](https://azure.microsoft.com/en-us/services/machine-learning-studio/)
- [comet.ml](https://www.comet.ml/)
- [SigOpt](https://sigopt.com/)

## Blog posts
- [A Conversation With Quoc Le: The AI Expert Behind Google AutoML](https://medium.com/syncedreview/a-conversation-with-quoc-le-the-ai-expert-behind-google-automl-73a7d0c9fe38)
- [fast.ai: An Opinionated Introduction to AutoML and Neural Architecture Search](https://www.fast.ai/2018/07/12/auto-ml-1/)
- [Introducing AdaNet: Fast and Flexible AutoML with Learning Guarantees](https://ai.googleblog.com/2018/10/introducing-adanet-fast-and-flexible.html)
- [Using Evolutionary AutoML to Discover Neural Network Architectures](https://ai.googleblog.com/2018/03/using-evolutionary-automl-to-discover.html)
- [Improving Deep Learning Performance with AutoAugment](https://ai.googleblog.com/2018/06/improving-deep-learning-performance.html)
- [AutoML for large scale image classification and object detection](https://ai.googleblog.com/2017/11/automl-for-large-scale-image.html)
- [Using Machine Learning to Discover Neural Network Optimizers](https://ai.googleblog.com/2018/03/using-machine-learning-to-discover.html)
- [Using Machine Learning to Explore Neural Network Architecture](https://ai.googleblog.com/2017/05/using-machine-learning-to-explore.html)

## Presentations
- [Advanced Machine Learning Day 3: Neural Architecture Search](https://www.youtube.com/watch?v=wL-p5cjDG64) by Debadeepta Dey (MSR)
- [Neural Architecture Search](https://www.youtube.com/watch?v=sROrvtXnT7Q&t=116s) by Quoc Le (Google Brain)

## Competitions, workshops and conferences
- [NIPS 2018 3rd AutoML Challenge: AutoML for Lifelong Machine Learning](http://automl.chalearn.org/)
- [AutoML Workshop in ICML](https://www.ml4aad.org/workshops/)

## Other curated resources on AutoML
- [Literature on Neural Architecture Search](https://www.ml4aad.org/automl/literature-on-neural-architecture-search/)
- [Awesome-AutoML-Papers](https://github.com/hibayesian/awesome-automl-papers)

# Practical applications
- [AutoML: Automating the design of machine learning models for autonomous driving](https://medium.com/waymo/automl-automating-the-design-of-machine-learning-models-for-autonomous-driving-141a5583ec2a) by Waymo

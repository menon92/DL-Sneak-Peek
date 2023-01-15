# Deep learning sneak peek
I try to explain various important terms of deep learning and machine learning. 
I will write this sort of tutorial for helping myself to build a clear understanding. 
If anyone get helped reading this It would be grateful for me

### Deep learning (`Basics`)
|Part|                             Topic                            | Git |  Colab |
|--- |--------------------------------------------------------------|---- |--------|
|01| টেন্সরফ্লোর প্রথমিক ধারনা, টেন্সর, কিছু ম্যাথ অপারেশন  | [link](./basics/dl_basics_part_1.ipynb) | x |
|02| গ্রাডিয়েন্ট-টেপ, হাইয়ার অডার ডেরিভেটিভ , গ্রাডিয়েন্ট-টেপের কিছু বিশেষ ব্যবহার | [link](./basics/dl_basics_part_2.ipynb) | x |
|03| লিনিয়ার রিগ্রেসন, প্লট লার্নিং কার্ভ | [link](./basics/dl_basics_part_3.ipynb) | x |
|04| সফটম্যাক্স ফাংশন কি এবং কেন কাজ করে | [link](./basics/dl_basics_part_4.ipynb) | x |
|05| মেশিন লার্নিংয়ে টাইপ - ১ ও টাইপ - ২ এরর কি ? | [link](./basics/type-1_vs_type_2_error.md) | x |

### Deep learning (`Computer vision`)
|Part|                             Topic                            | Git |  Colab |
|--- |--------------------------------------------------------------|---- |--------|
|01| ইমেজ ডাটা লোডিং, ফ্লাওয়ার ডাটাসেট, প্লট ইমেজ | [link](./vision/dl_computer_vision_part_1.ipynb) | [colab](https://bit.ly/2WEzPUe)|
|02| ইমেজ ডাটা পাইপলাইন, Keras ImageDataGenerator, ওয়ান-হট এনকোডিং | [link](./vision/dl_computer_vision_part_2.ipynb)|[colab](https://bit.ly/2ygaeHO)|
|03| ইমেজ ডাটা পাইপলাইন, tf.data , map ফাংশনের ব্যাবহার | [link](./vision/dl_computer_vision_part_3.ipynb) | [colab](https://bit.ly/2AF3zYy) | 
|04| tf.data ও ImageDataGenerator এর মধ্যে তুলনা, ট্রেনিং টাইম স্পীড আপ করা, ডাটা ক্যাশিং | [link](./vision/dl_computer_vision_part_4.ipynb)|[colab](https://bit.ly/2ZqZzW3)
|05| ইমেজ ডাটা পাইপলাইন tf.keras.utils.Sequence | [link](./vision/dl_computer_vision_part_5.ipynb) | [colab](https://colab.research.google.com/drive/12ACmzAawasOq_lEJU23s0DatOe8Qx9W2?usp=sharing)|
|07| ফ্লাওয়ার ইমেজ ক্লাসিফিকেশন, সাবক্লাস মডেল, tf.keras.utils.Sequence | [link](./vision/dl_computer_vision_part_6.ipynb)|[colab](https://colab.research.google.com/drive/11f0B03QGshbNCiUhmrlYggxoqfx1qkfk?usp=sharing)|
|08| কাস্টম লেয়ার কি ? কিভাবে কাস্টম লেয়ার  ডিজাইন করতে হয়| [link](./Writting_custom_layer.ipynb) | X |

### Audio
|Part|                             Topic                            |Git   |Colab    |
|--- |--------------------------------------------------------------|:----:|:--------:|
|01| অডিও ফাইল রিড, ফিচার(FT, STFT, Spectrogram, MFCC) ভিজুয়ালাইজ | [link](./audio/audio_feature_visualization.ipynb) | [colab](https://bit.ly/3VIZEP7) |
|02| স্পিচ টু টেক্সট সিস্টেম ডিজাইন এ কেন আমরা স্পেক্টোগ্রাম ব্যবহার করি? | [link](https://www.linkedin.com/post/edit/7019349990099603456/)| - |

### Deep learning (`NLP`)
|Part|                             Topic                            |Git   |Colab    |
|--- |--------------------------------------------------------------|:----:|:--------:|
|01|টোকেনাইজার কি এবং এটার বিভিন্ন প্রয়োগ | [link](./nlp/drive_into_tokenizer_part_1.ipynb) | [colab](https://bit.ly/31bh51g) |
|02|টোকেনাইজার এর আরও কিছু ব্যবহার | [lin k](./nlp/drive_into_tokenizer_part_2.ipynb) | [colab](https://bit.ly/3exnsjn) |

### ML Ops
|SL  |                             Topic                            | Link |
|--- |--------------------------------------------------------------|----- |
|01  | Rule of Machine Learning | [link](mlops/rules_of_ml.pdf) |
|02  | MLOPS 101: Tips, Tricks and Best Practices- Vladimir Osin PyData Eindhoven 2021 | [link](https://www.youtube.com/watch?v=dzSp3Zf897g)|
|03  | Hands on MLOps using CML tool | [link](https://www.youtube.com/playlist?list=PL7WG7YrwYcnDBDuCkFbcyjnZQrdskFsBz) |
|04  | ML development process | [link](mlops/ML-model-dev.drawio.png)

### ML data visualization
- [https://github.com/PAIR-code/facets](https://github.com/PAIR-code/facets)
- [Tensorboard guide for tensorflow 1.x](https://medium.com/analytics-vidhya/basics-of-using-tensorboard-in-tensorflow-1-2-b715b068ac5a)

### CNN
- [Machine Learning Foundations by google developer playlist](https://www.youtube.com/playlist?list=PLOU2XLYxmsII9mzQ-Xxug4l2o04JBrkLV)
- [Implement various CNN youtube playlist by MIT](https://t.co/RIbME80e06?amp=1)
- [CNN Architectures - implementations | MLT](https://github.com/Machine-Learning-Tokyo/CNN-Architectures/tree/master/Implementations)

### RNN
- [A friendly introduction to Recurrent Neural Networks](https://youtu.be/UNmqTiOnRfg)
- [The Attention Mechanism](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb)
- [attention-mechanism](https://blog.floydhub.com/attention-mechanism/)

### Transformers
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

### The loss functions
- [Transducer](https://lorenlugosch.github.io/posts/2020/11/transducer/)
- [Connectionist Temporal Classification (CTC)](https://distill.pub/2017/ctc/)

### NLP
- [Building models with tf.text (TF World '19)](https://youtu.be/iu_OSAg5slY)
- [Natural Language Processing (NLP) Zero to Hero - Play list](https://goo.gle/nlp-z2h)

### Datasets
- [Google dataset search](https://datasetsearch.research.google.com/)
- [The Big Bad NLP Database](https://datasets.quantumstat.com/)

### Courses
- [Intro to tensorflow for deeplearning (Udacity)](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187)
- [Full stack deeplearning](https://fullstackdeeplearning.com/course/)
- [Software engineering with ML (udacity)](https://www.udacity.com/course/aws-machine-learning-foundations--ud090)
- [Machine Learning Crash Course by Google](https://developers.google.com/machine-learning/crash-course/ml-intro)
- [TensorFlow, Keras and deep learning, without a PhD](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0)
- [Deep learing by stanford - CS230](https://stanford.edu/~shervine/teaching/cs-230/)
- [MIT deep learning lecture](http://introtodeeplearning.com/)
- [Intro to deep learning kaggle mini course](https://www.kaggle.com/learn/intro-to-deep-learning)
- [Kaggle free data science course](https://www.kaggle.com/learn/overview)

### Bangla ML/DL resource:
- [হাতেকলমে পাইথন ডিপ লার্নিং](https://github.com/raqueeb)
- [সহজ বাংলায় 'বাংলা' ন্যাচারাল ল্যাঙ্গুয়েজ প্রসেসিং (এনএলপি)](https://github.com/raqueeb/nlp_bangla)
- [বাংলায় মেশিন লার্নিং](https://ml.howtocode.com.bd/)
- [ডিপ লার্নিং ও আর্টিফিশিয়াল নিউরাল নেটওয়ার্ক](https://dl.howtocode.com.bd/)
- [বাংলায় ব্যাসিক ডাটা সায়েন্স শেখার কোর্স](https://ds.howtocode.com.bd/)

### Hyper params tunes
- [Effect of batch size on training dynamics](https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e)
- [Determining optimal batch size](https://towardsdatascience.com/how-to-break-gpu-memory-boundaries-even-with-large-batch-sizes-7a9c27a400ce)
- [Hyperparameter Importance | PyTorch Developer Day 2020](https://www.youtube.com/watch?v=jaPVoObpdO0&t=11515s)

### Machine learning in production
- [Rules of ML by google](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [5 Steps to take your model in production](https://towardsdatascience.com/take-your-machine-learning-models-to-production-with-these-5-simple-steps-35aa55e3a43c)
- [Building the Software 2 0 Stack (Andrej Karpathy)](https://www.youtube.com/watch?v=y57wwucbXR8&t=83s)
- [A Recipe for Training Neural Networks (Andrej Karpathy)](http://karpathy.github.io/2019/04/25/recipe/)
- [Engineering Practices for Software 2.0 (PyTorch Developer Conference)](https://www.youtube.com/watch?v=KJAnSyB6mME)
- [Machine Learning Interviews: Lessons from Both Sides - FSDL](https://docs.google.com/presentation/d/1MX2V6fTp71j1aztvY5HLYM44iLG4HYMrYd4Dxn6Cxnw/edit?usp=sharing)
- [Troubleshooting Deep Neural Networks](http://josh-tobin.com/assets/pdf/troubleshooting-deep-neural-networks-01-19.pdf)
---
- [Machine Learning System Design (Chip Huyen)](https://huyenchip.com/machine-learning-systems-design/toc.html)
- [Machine Learning Systems Design (Chip Huyen) CS 329S](https://stanford-cs329s.github.io/syllabus.html)
- [ML Interview book (chip Huyen)](https://huyenchip.com/ml-interviews-book/)
- [Real Time Machine Learning Challenges and Solution (Chip Huyen)](https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html)
- [Data Distributio Shifts and monittorig (Chip Huyen) ](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html)

### Miscellaneous
- [data-science-with-dl-nlp-advanced-techniques](https://www.kaggle.com/vbmokin/data-science-with-dl-nlp-advanced-techniques)
- [37-reasons-why-your-neural-network-is-not-working](https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607)
- [https://cs231n.github.io/](https://cs231n.github.io/)
- [tf-keras-rnn-ctc-example](https://chadrick-kwag.net/tf-keras-rnn-ctc-example/)
- [Keras debugging tips](https://keras.io/examples/keras_recipes/debugging_tips/)
- [TF2 mixed presition training speed up](https://github.com/sayakpaul/Mixed-Precision-Training-in-tf.keras-2.0/tree/master/With_Policy)
- [Effect of activation fucntion](https://www.youtube.com/watch?v=s-V7gKrsels)
- [tf2 object detection api](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
- [Eat tensorlfow in 30 days](https://github.com/lyhue1991/eat_tensorflow2_in_30_days)
- [Tensorflow examples](https://github.com/aymericdamien/TensorFlow-Examples)
- [TensorFlow Fall 2020 updates: Keynote & what’s new since TF2.2](https://www.youtube.com/watch?v=bUCciKeVx60&t=1395s)
- [Explainable Ai](https://ex.pegg.io/)
- [The NLP Index](https://index.quantumstat.com/)
- [Machine Learning Glossary](https://developers.google.com/machine-learning/glossary)
- [Beam search, How it works](https://towardsdatascience.com/foundations-of-nlp-explained-visually-beam-search-how-it-works-1586b9849a24)
- [Nvidia tao toolkit for developer]()https://developer.nvidia.com/tao-toolkit

---
### Books
- [Deep Learning with Python (Chollet) 2nd Edition](https://www.manning.com/books/deep-learning-with-python-second-edition)
- [Transfer Learning for NLP (Paul Azunre)](https://www.manning.com/books/transfer-learning-for-natural-language-processing)
- [Speech and Language Processing by Daniel Jurafsky](https://web.stanford.edu/~jurafsky/slp3/2.pdf)
- [Foundations of Statistical Natural Language Processing](https://nlp.stanford.edu/fsnlp/)
- [Dive into deep learning](https://d2l.ai/index.html)
- [Natural Language Processing with Transformers](https://learning.oreilly.com/library/view/natural-language-processing/9781098103231/)
- [Designing Machine Learning Systems](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/)


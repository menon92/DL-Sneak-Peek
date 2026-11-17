# AI ট্রেনিংয়ে GPU CPU RAM ও SSD এর ভূমিকা

## সূচিপত্র

- [১ মডেলের জন্য ডেটা কীভাবে প্রস্তুত হয়](#১-মডেলের-জন্য-ডেটা-কীভাবে-প্রস্তুত-হয়)
- [২ একটি অডিও মডেলের ট্রেনিং কীভাবে চলে](#২-একটি-অডিও-মডেলের-ট্রেনিং-কীভাবে-চলে)
- [৩ CPU এর কাজ ট্রেনিং শুরুর পরও থাকে](#৩-cpu-এর-কাজ-ট্রেনিং-শুরুর-পরও-থাকে)
- [৪ RAM কেন দরকার এবং কোন RAM এর কথা হচ্ছে](#৪-ram-কেন-দরকার-এবং-কোন-ram-এর-কথা-হচ্ছে)
- [৫ দ্রুত SSD কেন কাজে লাগে](#৫-দ্রুত-ssd-কেন-কাজে-লাগে)
- [৬ দ্রুত GPU থাকা সত্ত্বেও ট্রেনিং ধীর হওয়ার উদাহরণ](#৬-দ্রুত-gpu-থাকা-সত্ত্বেও-ট্রেনিং-ধীর-হওয়ার-উদাহরণ)
- [৭ হার্ডওয়্যার উন্নত করার আগে কী দেখব](#৭-হার্ডওয়্যার-উন্নত-করার-আগে-কী-দেখব)
- [তথ্যসূত্র](#তথ্যসূত্র)

আচ্ছা, AI মডেল তো GPU-তে ট্রেনিং হয়। তাহলে ফাস্টার হার্ডডিস্ক, ভালো CPU আর ফাস্টার RAM কেন দরকার?

কারণ, **GPU যে ডেটা নিয়ে হিসাব করবে, সেই ডেটা তার কাছে ঠিক সময়ে পৌঁছাতে হবে।** স্টোরেজ থেকে ডেটা পড়া, প্রয়োজনমতো প্রস্তুত করা, মেমরিতে রাখা এবং GPU-তে পাঠানোও পুরো ট্রেনিং প্রক্রিয়ার অংশ। এই কাজগুলো ধীর হলে শক্তিশালী GPU-ও ডেটার জন্য অপেক্ষা করে। [^9]

এখানে বড় deep learning মডেলের সাধারণ GPU training setup নিয়ে কথা বলছি। পুরো ব্যাপারটি বুঝতে আগে দেখি, ডেটা থেকে মডেল শেখা পর্যন্ত কী কী ঘটে।

## ১ মডেলের জন্য ডেটা কীভাবে প্রস্তুত হয়

আমাদের কাছে ডেটা থাকে টেক্সট, ছবি বা অডিও ফাইল হিসেবে। মডেলকে হিসাব করতে দেওয়ার জন্য ফাইলের তথ্য পড়ে সেটিকে **সেই মডেলের উপযোগী সংখ্যার বিন্যাসে** সাজাতে হয়। এই সংখ্যাগুলো সাধারণত tensor হিসেবে রাখা হয়। Tensor বলতে এখানে এক বা একাধিক মাত্রায় সাজানো সংখ্যার সংগ্রহ বোঝাচ্ছি।

| ডেটা | মডেলের জন্য প্রস্তুতির উদাহরণ |
| --- | --- |
| টেক্সট | Tokenizer দিয়ে লেখাকে token ID বা সংখ্যার ধারায় রূপান্তর করা। |
| ছবি | ছবির pixel value নেওয়া এবং প্রয়োজন অনুযায়ী resize ও normalization করা। |
| অডিও | ফাইল থেকে waveform পড়া; মডেল অনুযায়ী waveform, spectrogram বা অন্য উপস্থাপন তৈরি করা। |

সব মডেলের জন্য একটিই common format নেই। যেমন, কোনো অডিও মডেল waveform নেয়, আবার কোনোটি spectrogram নেয়। তাই “raw ডেটা দিয়ে ট্রেনিং হয় না” বলার চেয়ে “ডেটাকে মডেলের প্রত্যাশিত input-এ সাজাতে হয়” বলা বেশি নির্ভুল। [^1]

**এই প্রস্তুতির অনেক কাজ সাধারণত CPU করে, তবে সব কাজ CPU-তেই করতে হয় না।** কিছু decoding, image processing বা audio processing GPU-তেও করা যায়। কোন অংশ কোথায় চলবে, তা মডেল ও ব্যবহৃত software-এর ওপর নির্ভর করে। [^2]

## ২ একটি অডিও মডেলের ট্রেনিং কীভাবে চলে

ধরি, আমরা এমন একটি মডেল ট্রেনিং করছি, যেটি বাংলা অডিও শুনে লেখা তৈরি করবে। আমাদের কাছে অডিও এবং তার সঠিক transcript আছে। সাধারণভাবে একটি batch-এর জন্য কাজগুলো এমন হতে পারে। Batch মানে একসঙ্গে প্রসেস করার জন্য নেওয়া কয়েকটি উদাহরণ।

1. **ফাইল পড়া:** স্টোরেজ থেকে কয়েকটি অডিও ও তাদের transcript পড়া হয়।
2. **ডেটা প্রস্তুত করা:** প্রয়োজনমতো অডিও decode ও resample করা হয়। Transcript-কে token ID-তে রূপান্তর করা হয়।
3. **Batch তৈরি করা:** ভিন্ন দৈর্ঘ্যের অডিও ও লেখাকে মডেলের প্রয়োজন অনুযায়ী সাজানো হয়; দরকার হলে ছোটগুলোর সঙ্গে padding যোগ করা হয়।
4. **GPU-তে পাঠানো:** প্রস্তুত input ও training target সাধারণত system RAM থেকে GPU-এর মেমরিতে পাঠানো হয়। [^1][^3][^6]
5. **Prediction ও loss হিসাব:** মডেল input থেকে prediction করে। Training target-এর সঙ্গে তুলনা করে loss দিয়ে ভুলের পরিমাণ মাপা হয়।
6. **মডেল আপডেট করা:** Backpropagation দিয়ে gradient হিসাব করা হয়। Gradient নির্দেশ করে parameter বদলালে loss কীভাবে বদলাতে পারে। Optimizer সেই তথ্য ব্যবহার করে মডেলের parameter বা weight আপডেট করে। [^4]

তারপর পরের batch নিয়ে একই প্রক্রিয়া চলে। বড় মডেলে এই শেখার হিসাবের বড় অংশ GPU করে। সব training-এর জন্য মানুষের লেখা আলাদা label অবশ্য লাগে না; যেমন, next-token language model-এর target মূল লেখার পরের token থেকেই তৈরি করা যায়। [^10]

## ৩ CPU এর কাজ ট্রেনিং শুরুর পরও থাকে

ডেটা প্রস্তুতির কিছু কাজ আগেই করে রাখা যায়। যেমন, একবার text tokenization করে ফল ডিস্কে সংরক্ষণ করা যায়। এতে পরের training run-এ একই কাজ বারবার করতে হয় না। তবে আগে থেকে সব ফল রেখে দিলে স্টোরেজের প্রয়োজনও বাড়তে পারে।
আবার ফাইল পড়া, batch বানানো, padding করা বা প্রতিবার কিছু পরিবর্তন (augmentation) করে নতুন training example তৈরি করা ট্রেনিং চলার সময়ও লাগতে পারে। শেষের কাজটিকে augmentation বলা হয়; যেমন অডিওতে নিয়ন্ত্রিত noise যোগ করা। কোন কাজ আগে হবে আর কোনটি চলতে চলতে হবে, তা pipeline-এর নকশার ওপর নির্ভর করে।
ভালোভাবে সাজানো pipeline-এ GPU যখন বর্তমান batch নিয়ে কাজ করে, CPU তখন পরের batch প্রস্তুত করতে পারে। আগে থেকে পরের batch তৈরি করে রাখাকে **prefetching** বলে। এতে GPU-এর অপেক্ষা কমে। তবে CPU worker-এর সংখ্যা বেশি করলেই সবসময় গতি বাড়ে না; অতিরিক্ত worker RAM ও অন্যান্য resource-এর ওপর চাপ দিতে পারে। [^3][^9]

## ৪ RAM কেন দরকার এবং কোন RAM এর কথা হচ্ছে

**System RAM** হলো CPU-এর কাজের জায়গা। পড়া ডেটা, processing-এর মধ্যবর্তী ফল এবং প্রস্তুত batch এখানে রাখা যায়। RAM-এ ডেটা cache করে রাখলে একই তথ্যের জন্য বারবার ডিস্কে যেতে নাও হতে পারে। পুরো dataset একসঙ্গে RAM-এ রাখতে হয় না; প্রয়োজন অনুযায়ী অংশ অংশ করে পড়েও ট্রেনিং চলে। [^3][^7]
**GPU-এর VRAM** হলো GPU-এর নিজস্ব কাজের মেমরি। সাধারণ training setup-এ এতে model weight, চলতি batch, gradient, optimizer-এর তথ্য এবং হিসাবের মধ্যবর্তী ফল থাকে। তাই শুধু model file-এর সাইজ দেখে training-এর মোট GPU memory requirement বোঝা যায় না। [^5]

System RAM বাড়ালে GPU-এর VRAM সরাসরি বাড়ে না। দুটির ভূমিকা আলাদা। বিশেষ offloading বা shared-memory ব্যবস্থা থাকলেও সাধারণ dedicated GPU server-এর ক্ষেত্রে এই পার্থক্যটি মনে রাখা দরকার। [^5][^6]

RAM-এর ক্ষেত্রেও দুটি বিষয় আলাদা:
- **Capacity:** কতটা ডেটা রাখা যায়। পর্যাপ্ত RAM থাকলে batch, buffer ও cache ধরে রাখা সহজ হয়। RAM কম পড়ে OS যদি ডেটা বারবার ডিস্কে সরিয়ে আবার পড়ে, অর্থাৎ swapping হয়, তাহলে গতি অনেক কমতে পারে।
- **Bandwidth:** কত দ্রুত RAM থেকে ডেটা পড়া বা RAM-এ লেখা যায়। CPU-এর data processing যদি এই গতির ওপর নির্ভর করে, তাহলে বেশি memory bandwidth কাজে লাগতে পারে।

তবে RAM-এর advertised speed বাড়লেই training একই অনুপাতে দ্রুত হবে না। CPU, storage, GPU-তে data transfer অথবা GPU-এর নিজের হিসাবের সময়—অন্য কোনোটি সীমা তৈরি করতে পারে। [^6][^7]

## ৫ দ্রুত SSD কেন কাজে লাগে

Dataset সাধারণত SSD, HDD বা দূরের কোনো storage-এ থাকে। সেখান থেকে সময়মতো ডেটা পড়তে না পারলে CPU পরের batch প্রস্তুত করতে পারে না; তার প্রভাব শেষ পর্যন্ত GPU-তে পড়ে।
অনেক ছোট ফাইল থেকে ডেটা পড়ার ক্ষেত্রে শুধু প্রতি সেকেন্ডে কত GB পড়া যায়, সেটিই যথেষ্ট তথ্য নয়। ফাইল খুলতে ও ছোট ছোট read request শেষ করতে কত সময় লাগছে, সেটিও গুরুত্বপূর্ণ। আবার দূরের storage থেকে পড়লে network-এর গতিও সীমা তৈরি করতে পারে। [^11]
স্টোরেজে শুধু training data পড়া হয় না; মাঝেমধ্যে **checkpoint**-ও লেখা হয়। Checkpoint হলো ট্রেনিংয়ের অগ্রগতি সংরক্ষণ করা। এতে model weight-এর পাশাপাশি optimizer-এর অবস্থা ও আবার শুরু করার প্রয়োজনীয় তথ্য থাকতে পারে। Checkpoint লেখা যদি training-কে অপেক্ষা করায়, দ্রুত storage সেই অপেক্ষা কমাতে সাহায্য করে। [^8]
তবে dataset-এর প্রয়োজনীয় অংশ RAM cache থেকেই পাওয়া গেলে, অথবা GPU-এর হিসাবেই সবচেয়ে বেশি সময় গেলে, আরও দ্রুত SSD বসিয়ে বড় speedup নাও পাওয়া যেতে পারে। কোথায় সময় যাচ্ছে, সেটিই আগে দেখতে হবে।

## ৬ দ্রুত GPU থাকা সত্ত্বেও ট্রেনিং ধীর হওয়ার উদাহরণ

নিচের সংখ্যাগুলো বোঝানোর জন্য কাল্পনিক; এগুলো কোনো hardware benchmark নয়। ধরি, GPU-এর এক batch নিয়ে শেখার হিসাব করতে লাগে ১০০ মিলিসেকেন্ড। কিন্তু input pipeline গড়ে প্রতি ৩০০ মিলিসেকেন্ডে একটি নতুন batch প্রস্তুত করে GPU-তে পৌঁছাতে পারে।
GPU-এর কাজ এবং পরের batch-এর প্রস্তুতি পাশাপাশি চললেও, pipeline চালু হয়ে যাওয়ার পর গড়ে ৩০০ মিলিসেকেন্ডে একটি batch শেষ হবে। GPU প্রায় ১০০ মিলিসেকেন্ড কাজ করে বাকি ২০০ মিলিসেকেন্ড নতুন ডেটার জন্য অপেক্ষা করবে।

| অবস্থা | নতুন batch পৌঁছানোর ব্যবধান | GPU-এর হিসাবের সময় | গড়ে একটি batch শেষ হওয়ার ব্যবধান |
| --- | ---: | ---: | ---: |
| ধীর input pipeline | ৩০০ ms | ১০০ ms | প্রায় ৩০০ ms |
| উন্নত input pipeline | ৮০ ms | ১০০ ms | প্রায় ১০০ ms |

দ্বিতীয় অবস্থায় input pipeline যথেষ্ট দ্রুত হওয়ায় GPU-কে প্রায় আর অপেক্ষা করতে হয় না। এই সরল উদাহরণে একই GPU দিয়ে প্রায় তিন গুণ batch প্রসেস করা সম্ভব। এখানে পর্যাপ্ত buffering ও কাজের overlap ধরা হয়েছে; শুরুর সময় এবং অন্যান্য overhead বাদ দেওয়া হয়েছে।
এরপর input pipeline আরও দ্রুত করলেও training একই অনুপাতে বাড়বে না, কারণ তখন GPU-এর ১০০ মিলিসেকেন্ড হিসাবের সময়ই গতি নির্ধারণ করছে।

## ৭ হার্ডওয়্যার উন্নত করার আগে কী দেখব

যে অংশের জন্য পরের কাজ আটকে থাকে, সেটিই bottleneck। কোনটি bottleneck, তা workload অনুযায়ী বদলায়।

- ডেটা পড়তে দেরি হলে storage, file layout ও network দেখব।
- ডেটা decode বা প্রস্তুত করতে দেরি হলে CPU, processing code ও worker সংখ্যা দেখব।
- RAM কম পড়লে buffer, cache, worker-এর memory usage এবং RAM capacity দেখব।
- ডেটা প্রস্তুত থাকার পরও GPU-তে পৌঁছাতে দেরি হলে transfer ও সংযোগের সীমা দেখব।
- GPU নিয়মিত ডেটা পেয়েও হিসাব শেষ করতে বেশি সময় নিলে model computation ও GPU-এর সক্ষমতা দেখব। [^6][^9]

একাধিক GPU যোগ করলে তাদের সবার জন্য পর্যাপ্ত গতিতে ডেটা সরবরাহ করতে হবে। তাই শুধু GPU সংখ্যা বাড়িয়ে সমান হারে training speed বাড়বে—এমন নিশ্চয়তা নেই।

**দ্রুত training-এর জন্য GPU, CPU, RAM, storage এবং data pipeline-এর মধ্যে সামঞ্জস্য দরকার।** GPU কোথায় অপেক্ষা করছে, সেটা মেপে সেই অংশ উন্নত করলেই hardware-এর সক্ষমতা ভালোভাবে কাজে লাগানো যায়।

## তথ্যসূত্র

[^1]: [Hugging Face — Preprocess](https://huggingface.co/docs/transformers/main/en/preprocessing)
[^2]: [NVIDIA — DALI Documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)
[^3]: [PyTorch — torch.utils.data](https://docs.pytorch.org/docs/stable/data.html)
[^4]: [PyTorch — Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
[^5]: [Hugging Face — Model training anatomy](https://huggingface.co/docs/transformers/v4.37.0/en/model_memory_anatomy)
[^6]: [NVIDIA — CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
[^7]: [NVIDIA — Tips on Scaling Storage for AI Training and Inferencing](https://developer.nvidia.com/blog/tips-on-scaling-storage-for-ai-training-and-inferencing/)
[^8]: [PyTorch — Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
[^9]: [PyTorch — Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
[^10]: [Hugging Face — Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling)
[^11]: [NVIDIA — Storage Performance Basics for Deep Learning](https://developer.nvidia.com/blog/storage-performance-basics-for-deep-learning/)

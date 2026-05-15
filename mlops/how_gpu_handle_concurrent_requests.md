# GPU কিভাবে Concurrent Request প্রসেস করে এবং LLM সারভিং এ In-flight Batching কেন এতো গুরুত্বপূর্ণ

আজকের AI application, API server, image classifier বা LLM chatbot—সব জায়গাতেই একটা common challenge থাকে:

> একই সময়ে অনেক user request এলে system কীভাবে সেগুলো efficiently handle করবে?

আমরা অনেক সময় বলি, “CPU concurrent request handle করে” বা “GPU concurrent request handle করে।” কিন্তু technically CPU বা GPU সরাসরি HTTP request বোঝে না। Request আসে application server-এ। তারপর operating system, runtime, scheduler, queue, worker thread, event loop এবং inference server মিলে সেই request-গুলোকে CPU বা GPU কাজ হিসেবে execute করে।

এই লেখায় আমরা সহজভাবে দেখবো:

1. CPU কীভাবে concurrent request handle করে
2. GPU কীভাবে concurrent request process করে
3. GPU serving-এ `batching` কেন দরকার
4. LLM serving-এ `in-flight batching` বা continuous batching কীভাবে কাজ করে

## Table of contents

- [1. Concurrent request বলতে আসলে কী বোঝায়?](#1-concurrent-request-বলতে-আসলে-কী-বোঝায়)
- [2. CPU কীভাবে concurrent request manage করে?](#2-cpu-কীভাবে-concurrent-request-manage-করে)
- [3. CPU concurrency-এর তিনটি common model](#3-cpu-concurrency-এর-তিনটি-common-model)
  - [Model 1: Thread per request](#model-1-thread-per-request)
  - [Model 2: Worker pool](#model-2-worker-pool)
  - [Model 3: Event loop / async I/O](#model-3-event-loop--async-io)
- [4. CPU scheduling সহজ ভাষায়](#4-cpu-scheduling-সহজ-ভাষায়)
- [5. CPU concurrency কোথায় ভালো কাজ করে?](#5-cpu-concurrency-কোথায়-ভালো-কাজ-করে)
- [6. GPU কীভাবে concurrent request handle করে?](#6-gpu-কীভাবে-concurrent-request-handle-করে)
- [7. GPU concurrency বনাম CPU concurrency](#7-gpu-concurrency-বনাম-cpu-concurrency)
- [8. GPU-তে multiple request আলাদা আলাদা চালালে সমস্যা কী?](#8-gpu-তে-multiple-request-আলাদা-আলাদা-চালালে-সমস্যা-কী)
- [9. Batching কী?](#9-batching-কী)
- [10. Batching-এর সহজ উদাহরণ](#10-batching-এর-সহজ-উদাহরণ)
- [11. Dynamic batching কী?](#11-dynamic-batching-কী)
- [12. Dynamic batching-এর tradeoff](#12-dynamic-batching-এর-tradeoff)
- [13. GPU concurrent model execution](#13-gpu-concurrent-model-execution)
- [14. LLM serving কেন আলাদা?](#14-llm-serving-কেন-আলাদা)
- [15. Static batching problem in LLM](#15-static-batching-problem-in-llm)
- [16. In-flight batching কী?](#16-in-flight-batching-কী)
- [17. In-flight batching-এর সহজ analogy](#17-in-flight-batching-এর-সহজ-analogy)
- [18. In-flight batching timeline example](#18-in-flight-batching-timeline-example)
- [19. In-flight batching LLM generation loop](#19-in-flight-batching-llm-generation-loop)
- [20. KV cache কেন গুরুত্বপূর্ণ?](#20-kv-cache-কেন-গুরুত্বপূর্ণ)
- [21. In-flight batching-এ prefill ও decode একসাথে](#21-in-flight-batching-এ-prefill-ও-decode-একসাথে)
- [22. Dynamic batching বনাম in-flight batching](#22-dynamic-batching-বনাম-in-flight-batching)
- [23. End-to-end serving architecture](#23-end-to-end-serving-architecture)
- [24. Practical example: Chatbot serving](#24-practical-example-chatbot-serving)
- [25. কি কি Latency metrics বুঝতে হবে এবং মনিটর করতে হবে](#25-latency-metrics-বুঝতে-হবে)
- [26. Batching-এর ভুল ব্যবহার](#26-batching-এর-ভুল-ব্যবহার)
  - [Problem 1: Batch delay বেশি](#problem-1-batch-delay-বেশি)
  - [Problem 2: Batch size খুব বড়](#problem-2-batch-size-খুব-বড়)
  - [Problem 3: Long request short request-কে block করে](#problem-3-long-request-short-request-কে-block-করে)
  - [Problem 4: CPU bottleneck](#problem-4-cpu-bottleneck)
- [27. কখন কোন strategy ব্যবহার করবেন?](#27-কখন-কোন-strategy-ব্যবহার-করবেন)
  - [CPU async/event loop ব্যবহার করুন যখন:](#cpu-asyncevent-loop-ব্যবহার-করুন-যখন)
  - [CPU worker pool ব্যবহার করুন যখন:](#cpu-worker-pool-ব্যবহার-করুন-যখন)
  - [GPU dynamic batching ব্যবহার করুন যখন:](#gpu-dynamic-batching-ব্যবহার-করুন-যখন)
  - [GPU in-flight batching ব্যবহার করুন যখন:](#gpu-in-flight-batching-ব্যবহার-করুন-যখন)
- [28. Final mental model](#28-final-mental-model)
- [29. One-line summary](#29-one-line-summary)

---

## 1. Concurrent request বলতে আসলে কী বোঝায়?

ধরুন আপনার API server-এ একই সময়ে ১,০০০ user request এসেছে।

প্রতিটি request হয়তো এমন কাজ করছে:

```text
User request
→ authentication
→ database read
→ preprocessing
→ model inference
→ response
```

এই request-গুলোর সব কাজ একই ধরনের নয়।

কিছু কাজ CPU করে:

```text
JSON parsing
Tokenization
Database response handling
Business logic
Post-processing
```

কিছু কাজ GPU করে:

```text
Neural network inference
Matrix multiplication
Attention computation
Token generation
Embedding computation
```

তাই concurrent request serving হলো CPU, GPU, memory, network, queue এবং scheduler—সবকিছুর coordination problem।

---

## 2. CPU কীভাবে concurrent request manage করে?

CPU সাধারণত কয়েকটি core নিয়ে তৈরি হয়। যেমন ৮-core CPU মানে CPU একই সময়ে বাস্তবে ৮টি heavy computation task parallel-এ চালাতে পারে। কিন্তু web server অনেক সময় হাজার হাজার request handle করতে পারে, কারণ সব request একসাথে CPU ব্যবহার করে না।

অনেক request আসলে অপেক্ষা করে:

```text
Waiting for database
Waiting for network
Waiting for disk
Waiting for another service
```

এই wait time-এর সময় CPU অন্য request-এর কাজ করতে পারে।

---

## 3. CPU concurrency-এর তিনটি common model

### Model 1: Thread per request

এখানে প্রতিটি request-এর জন্য একটি thread assign করা হয়।

```text
Request A → Thread 1
Request B → Thread 2
Request C → Thread 3
Request D → Thread 4
```

Thread কাজ করে। যদি database-এর জন্য wait করতে হয়, thread block হয়ে যায়।

এটা বোঝা সহজ, কিন্তু হাজার হাজার request এলে thread creation, memory usage এবং context switching overhead বেড়ে যায়।

---

### Model 2: Worker pool

এখানে fixed number of worker থাকে।

```text
Incoming requests → Queue → Worker 1
                         → Worker 2
                         → Worker 3
                         → Worker 4
```

Request বেশি হলে queue-তে wait করে। Worker free হলে next request নেয়।

এটা predictable এবং production system-এ common।

---

### Model 3: Event loop / async I/O

এখানে একটি বা অল্প কয়েকটি thread অনেক request manage করে।

ধরুন request A database call করলো। CPU বসে থাকে না। A-কে wait state-এ রেখে CPU request B, C, D-এর কাজ করে।

```text
Request A → DB wait
Request B → process
Request C → network wait
Request D → process
```

Python-এর `asyncio` event loop asynchronous tasks, callbacks, network I/O এবং subprocess চালায়। ([Python Event Loop][1])

---

## 4. CPU scheduling সহজ ভাষায়

ধরুন CPU core একটি দোকানের cashier-এর মতো।

Cashier একই সময়ে একজনের bill process করতে পারে। কিন্তু line-এ অনেক customer আছে। Operating system scheduler ঠিক করে কে এখন CPU পাবে, কে wait করবে।

Linux-এর Completely Fair Scheduler বা CFS runnable task-গুলোর মধ্যে CPU time fair ভাবে ভাগ করার চেষ্টা করে। CFS task-এর virtual runtime track করে এবং যে task তুলনামূলকভাবে কম CPU time পেয়েছে তাকে আগে চালানোর চেষ্টা করে। ([Linux CFS Scheduler][2])

সহজভাবে:

```text
CPU core = cashier
Request/thread = customer
OS scheduler = line manager
```

CPU concurrency মানে সব request literally একই সময়ে execute হচ্ছে না। বরং CPU খুব দ্রুত task switch করে এবং I/O wait-এর সময় অন্য কাজ করে।

---

## 5. CPU concurrency কোথায় ভালো কাজ করে?

CPU concurrent request handling ভালো কাজ করে যখন workload হয়:

```text
I/O-bound workload:
- database call
- network API call
- file read/write
- cache lookup
```

কিন্তু workload যদি CPU-bound হয়, যেমন:

```text
large JSON compression
video encoding
heavy encryption
complex calculation
```

তাহলে CPU core যত, parallel heavy work ততটাই realistically চালানো যায়। ৮-core CPU-তে ১০,০০০ CPU-bound task দিলে তারা queue-তে wait করবে বা context switching overhead তৈরি করবে।

---

# 6. GPU কীভাবে concurrent request handle করে?

GPU, CPU-এর মতো general-purpose request manager নয়। GPU অনেক ছোট ছোট core বা streaming multiprocessor ব্যবহার করে massive parallel computation করে। GPU সবচেয়ে ভালো কাজ করে যখন একই ধরনের mathematical operation অনেক data-এর উপর একসাথে চালানো হয়।

Deep learning model inference-এ মূল কাজগুলো হলো:

```text
Matrix multiplication
Vector operation
Attention computation
Activation function
Normalization
```

এগুলো GPU-এর জন্য perfect workload।

কিন্তু একটি গুরুত্বপূর্ণ বিষয় আছে:

> GPU অনেক request আলাদা আলাদা ছোট কাজ হিসেবে পেলে অনেক সময় inefficient হয়। GPU বড় batch পেলে বেশি efficient হয়।

---

## 7. GPU concurrency বনাম CPU concurrency

CPU concurrency সাধারণত request-level:

```text
Request A → Thread A
Request B → Thread B
Request C → Thread C
```

GPU concurrency সাধারণত computation-level:

```text
Batch of requests
→ large tensor
→ CUDA kernels
→ parallel execution on GPU
```

CUDA programming model-এ `host computation` (CPU-তে প্রসেসিং), `device computation` (GPU-তে প্রসেসিং) এবং `memory transfer` (CPU \(\leftrightarrow \) GPU ডাটা ট্রান্সফার) independent task হিসেবে concurrently operate করতে পারে। [CUDA streams][8] ব্যবহার করে command sequence তৈরি করা যায়; একই stream-এর commands order অনুযায়ী চলে, আর different streams-এর commands out-of-order বা concurrently execute হতে পারে। ([NVIDIA Docs][3])

সহজভাবে:

```text
CPU:
Many different tasks, flexible switching

GPU:
Few large mathematical jobs, massive parallel execution
```

---

## 8. GPU-তে multiple request আলাদা আলাদা চালালে সমস্যা কী?

ধরুন ৪ জন user image classification request পাঠিয়েছে।

Without batching:

```text
Request A → GPU run
Request B → GPU run
Request C → GPU run
Request D → GPU run
```

এখানে প্রতিটি request ছোট হলে GPU পুরোপুরি busy নাও হতে পারে। Kernel launch overhead থাকে, memory transfer overhead থাকে, আর GPU utilization কম থাকে।

এটা অনেকটা বড় বাসে একজন করে passenger নিয়ে বগুড়া থেকে চট্টগ্রাম যাত্রা করার মতো।

---

## 9. Batching কী?

Batching হলো একাধিক request একসাথে করে model-এ পাঠানো।

With batching:

```text
Request A
Request B
Request C
Request D
     ↓
Batch [A, B, C, D]
     ↓
GPU run once
```

Deep learning model input tensor-এ সাধারণত batch dimension থাকে:

```text
Single request:
[1, input_size]

Batch request:
[4, input_size]
```

GPU একই model computation একসাথে ৪টি input-এর উপর চালায়। এতে throughput বাড়ে।

---

## 10. Batching-এর সহজ উদাহরণ

ধরুন আপনার কাছে একটি image classifier আছে।

প্রতিটি image আলাদা চালালে:

```text
Image 1 → 10 ms
Image 2 → 10 ms
Image 3 → 10 ms
Image 4 → 10 ms

Total = 40 ms
```

Batch করে চালালে:

```text
[Image 1, Image 2, Image 3, Image 4] → 16 ms

Total = 16 ms
```

এখানে single request latency কিছু ক্ষেত্রে একটু বাড়তে পারে, কারণ request batch তৈরি হওয়ার জন্য সামান্য wait করে। কিন্তু total throughput অনেক বাড়ে।

---

# 11. Dynamic batching কী?

Static batching মানে আগে থেকেই fixed batch বানানো।

Dynamic batching মানে server runtime-এ incoming request দেখে batch তৈরি করে।

Dynamic batching হলো এমন feature যেখানে server এক বা একাধিক inference request combine করে dynamically batch তৈরি করে throughput maximize করে। Scheduler চাইলে সামান্য delay রাখতে পারে যাতে আরও request batch-এ join করতে পারে। ([Here is How NVIDIA Triton Server Dynamic Batching works][4])

Example:

```text
t = 0 ms: Request A arrives
t = 1 ms: Request B arrives
t = 2 ms: Request C arrives
t = 3 ms: Request D arrives

If we configdure scheduler waits max 5 ms then
Batch = [A, B, C, D]
GPU runs batch
```

---

## 12. Dynamic batching-এর tradeoff

Dynamic batching-এর মূল tradeoff:

```text
Bigger batch → higher throughput
More waiting → higher latency
```

তাই production system-এ সাধারণত config থাকে:

```text
max_batch_size
preferred_batch_size
max_queue_delay
queue_timeout
priority
```

যদি `max_queue_delay` খুব বেশি হয়, user বেশি wait করবে।
যদি খুব কম হয়, GPU ছোট batch পাবে এবং GPU utilization কম হতে পারে।

---

# 13. GPU concurrent model execution

Batching ছাড়াও GPU serving system একই model-এর multiple instance বা multiple model parallel চালাতে পারে।

Nvidia Triton Inferance Server একই system-এ multiple model অথবা একই model-এর multiple instance parallel execute করতে পারে। তবে same model-এর multiple request default অবস্থায় serialize হতে পারে; model configuration-এ instance group ব্যবহার করে একই model-এর parallel execution instance বাড়ানো যায়। এবং এটার আরেকটা সুবিধা হল মডেলের ওয়েট শেয়ার করতে পারে ফলে instance group বাড়ালেও কম GPU মেমরি ব্যবহার হয়। ([Concurrent Model Execution][5])

Example:

```text
GPU
├── Model instance 1 → Batch A
├── Model instance 2 → Batch B
└── Model instance 3 → Batch C
```

এটা useful যখন একটি batch GPU পুরোপুরি saturate করে না, অথবা multiple model serve করতে হয়।

---

# 14. LLM serving কেন আলাদা?

Image classification বা embedding model সাধারণত একবার forward pass করে response দেয়।

কিন্তু LLM response generate করে token-by-token:

```text
Prompt: "Explain batching"
Output:
Token 1 → "Batching"
Token 2 → "is"
Token 3 → "a"
Token 4 → "technique"
...
```

LLM inference-এর দুটি major phase থাকে:

```text
1. Prefill / context phase
   Prompt tokens process করা

2. Decode / generation phase
   একবারে একটি করে output token generate করা
```

এই token-by-token generation-এর কারণে normal batching `inefficient` হতে পারে।

---

## 15. Static batching problem in LLM

ধরুন তিনটি request একসাথে batch হলো।

```text
Request A → needs 8 output tokens
Request B → needs 2 output tokens
Request C → needs 6 output tokens
```

Static batch হলে batch lockstep-এ চলতে পারে:

```text
Step 1: A B C
Step 2: A B C # Generation of 2nd token of B done
Step 3: A - C
Step 4: A - C
Step 5: A - C
Step 6: A - C # Generation of 6th token of B done
Step 7: A - -
Step 8: A - -
```

Request B মাত্র ২ token-এর পর শেষ, কিন্তু batch structure-এর কারণে অনেক compute slot waste হতে পারে।

TensorRT-LLM executor এ static batching-কে এমন scheme হিসেবে describe করা হয়েছে যেখানে requests lockstep-এ চলে এবং batch-এর request-গুলো maximum input/output sequence length পর্যন্ত padded হয়। ([TRT LLM Executor][6])

> STATIC refers to the traditional batching scheme with a batch of requests running in lockstep until the full generation for all of them is complete. Requests in a batch are all padded up to the maximum input and output sequence length of any member of the batch

---

# 16. In-flight batching কী?

In-flight batching, continuous batching বা iteration-level batching হলো LLM serving-এর জন্য বেশি advanced batching technique।

এখানে batch একবার তৈরি হয়ে fixed থাকে না। বরং generation loop চলাকালীন প্রতিটি iteration-এ scheduler দেখে:

```text
কোন request শেষ হয়েছে?
কোন নতুন request এসেছে?
GPU memory/KV cache space আছে?
নতুন request active batch-এ ঢোকানো যাবে?
```

TensorRT-LLM documentation অনুযায়ী in-flight batching-এ newly arrived requests চলমান batch-এ dynamically incorporate করা হয় এবং কোনো request end condition meet করলে padding ছাড়া return করা হয়। ([TRT LLM Executor][6])

> INFLIGHT refers to a scheme where newly arrived requests are dynamically incorporated into the batch under execution, and requests are returned as soon as the end condition is met without any padding
---

## 17. In-flight batching-এর সহজ analogy

একটি বাস station থেকে ছাড়লো।

Normal batching:

```text
Bus leaves.
New passenger arrives.
Passenger must wait for next bus.
```

In-flight batching:

```text
Bus is still moving slowly.
A passenger arrives.
Conductor lets them enter.
Another passenger gets down.
Seat becomes free.
New passenger takes that seat.
```

LLM serving-এ seat মানে GPU batch capacity বা KV cache capacity।

---

# 18. In-flight batching timeline example

ধরুন active batch-এ A, B, C আছে।

```text
Iteration 1:
[A, B, C] → next token

Iteration 2:
[A, B, C] → next token
B finishes

Iteration 3:
[A, C] + new request D joins
[A, C, D] → next token

Iteration 4:
[A, C, D] → next token
C finishes

Iteration 5:
[A, D] + new request E joins
[A, D, E] → next token
```

এখানে GPU খালি বসে নেই। ছোট response শেষ হলে তার জায়গায় নতুন request ঢুকে যাচ্ছে।

---

## 19. In-flight batching LLM generation loop

Simplified pseudocode:

```python
active_batch = []

while server_is_running:
    new_requests = queue.get_ready_requests()

    for request in new_requests:
        if has_kv_cache_space(request):
            active_batch.add(request)

    run_one_generation_iteration(active_batch)

    for request in active_batch:
        if request.is_finished():
            send_response(request)
            active_batch.remove(request)
            free_kv_cache(request)
```

এই loop প্রতি token step বা scheduling iteration-এ চলতে পারে।

---

# 20. KV cache কেন গুরুত্বপূর্ণ?

LLM প্রতিটি নতুন token generate করার সময় আগের tokens-এর attention information reuse করে। এই stored information-কে KV cache বলা হয়।

KV cache ছাড়া model-কে বারবার পুরো sequence recompute করতে হতো। কিন্তু KV cache GPU memory খায়। তাই in-flight batching scheduler শুধু batch size দেখে না; সে দেখে GPU memory এবং KV cache space আছে কিনা।

Generation phase-এ past K এবং V elements cache হিসেবে রাখা হয়, এবং TensorRT-LLM-এ প্রতি Transformer layer-এর জন্য KV cache থাকে। Paged KV cache block আকারে cache distribute ও recycle করা হয়। ([Multi-Head, Multi-Query, and Group-Query Attention with KV cache][7])

---

## 21. In-flight batching-এ prefill ও decode একসাথে

LLM request দুই phase-এ থাকে:

```text
Prefill: prompt process
Decode: token generation
```

Traditional batching অনেক সময় prefill আর decode আলাদা ভাবে চালায়। কিন্তু in-flight batching এগুলো smarter ভাবে interleave করতে পারে।

in-flight batching context phase-এর sequence এবং generation phase-এর sequence একসাথে process করতে পারে, যাতে request interleave হয়, latency কমে এবং GPU ভালোভাবে ব্যবহার হয়। ([Multi-Head, Multi-Query, and Group-Query Attention with KV cache][7])

Example:

```text
Batch at one iteration:

Request A → decoding next token
Request B → decoding next token
Request C → prefill prompt chunk
Request D → decoding next token
```

এতে GPU compute capacity আরও ভালোভাবে ব্যবহার করা যায়।

---

# 22. Dynamic batching বনাম in-flight batching

| বিষয়              | Dynamic batching                                  | In-flight batching                               |
| ----------------- | ------------------------------------------------- | ------------------------------------------------ |
| Common use case   | Image model, embedding model, `stateless inference` | LLM text generation, `statefull inference` (Autoregressive / Causal model)|
| Batch কখন তৈরি হয় | Inference call-এর আগে                             | Generation চলাকালীন update হয়                    |
| Request শেষ হলে   | পুরো batch execution শেষ হয়                       | Request সাথে সাথে বের হতে পারে                   |
| নতুন request      | সাধারণত next batch-এ যায়                          | active batch-এ join করতে পারে                    |
| Padding issue     | থাকতে পারে                                        | কমানো যায়                                        |
| Main goal         | Throughput বাড়ানো                                 | Throughput + latency + GPU utilization উন্নত করা |

---

# 23. End-to-end serving architecture

একটি modern AI serving system দেখতে এমন হতে পারে:

```text
Client requests
      ↓
API Gateway / Load Balancer
      ↓
CPU server
- auth
- validation
- tokenization
- queue management
      ↓
Inference scheduler
- batching
- priority
- timeout
- KV cache tracking
      ↓
GPU
- model forward pass
- prefill
- decode
      ↓
CPU post-processing
      ↓
Response to user
```

এখানে CPU request orchestration করে। GPU heavy model computation করে।

---

# 24. Practical example: Chatbot serving

ধরুন আপনার chatbot server-এ একই সময়ে ৫টি request এলো।

```text
A: "Hi"                         → short prompt, short output
B: "Write a blog post"           → short prompt, long output
C: "Summarize this document..."  → long prompt, medium output
D: "Translate this sentence"     → short prompt, short output
E: "Explain Kubernetes"          → short prompt, medium output
```

Static batching করলে সমস্যা:

```text
B long output generate করছে
A ও D দ্রুত শেষ হলেও batch slot waste হতে পারে
C-এর prompt long হওয়ায় others wait করতে পারে
```

In-flight batching করলে:

```text
A finishes → slot freed
D finishes → slot freed
New request F joins
B continues
C continues
E continues
GPU remains busy
```

এটাই high-throughput LLM serving-এর core idea।

---

# 25. কি কি Latency metrics বুঝতে হবে এবং মনিটর করতে হবে

GPU batching করলে শুধু average latency দেখা যথেষ্ট নয়। Production serving-এ কয়েকটি metric important:

```text
RPS / QPS:
প্রতি সেকেন্ডে কত request serve হচ্ছে

Tokens/sec:
প্রতি সেকেন্ডে কত token generate হচ্ছে

TTFT:
Time To First Token

TPOT:
Time Per Output Token

P95 / P99 latency:
Slow users কত delay পাচ্ছে

Queue time:
Request GPU-তে যাওয়ার আগে কতক্ষণ wait করছে

GPU utilization:
GPU কতটা busy

KV cache usage:
GPU memory-এর কতটা KV cache হিসেবে ব্যবহৃত হচ্ছে
```

LLM chatbot-এ TTFT খুব important, কারণ user প্রথম token দ্রুত দেখতে চায়। Long generation workload-এ tokens/sec এবং TPOT বেশি important।

---

# 26. Batching-এর ভুল ব্যবহার

Batching সবসময় ভালো নয়। ভুল configuration করলে latency খারাপ হতে পারে।

### Problem 1: Batch delay বেশি

```text
max_queue_delay = 100 ms
```

এতে throughput বাড়তে পারে, কিন্তু user first response পেতে দেরি করবে।

### Problem 2: Batch size খুব বড়

বড় batch, GPU memory বেশি খায়। LLM-এ KV cache দ্রুত full হয়ে যেতে পারে।

### Problem 3: Long request short request-কে block করে

যদি scheduler smart না হয়, long prompt বা long output ছোট request-এর latency বাড়িয়ে দিতে পারে।

### Problem 4: CPU bottleneck

GPU powerful হলেও CPU tokenization, JSON processing, network response, logging, compression বা database call bottleneck হতে পারে।

---

# 27. কখন কোন strategy ব্যবহার করবেন?

### CPU async/event loop ব্যবহার করুন যখন:

```text
- request অনেক
- কাজ I/O-bound
- database/network wait বেশি
- প্রতি request CPU computation কম
```

### CPU worker pool ব্যবহার করুন যখন:

```text
- controlled concurrency দরকার
- blocking library ব্যবহার করছেন
- predictable queueing চান
```

### GPU dynamic batching ব্যবহার করুন যখন:

```text
- stateless model
- image classification
- embedding generation
- recommendation model
- fixed or similar input shape
```

### GPU in-flight batching ব্যবহার করুন যখন:

```text
- LLM serving করছেন
- output length variable
- prompt length variable
- chat completion / text generation
- high throughput দরকার
- GPU utilization improve করতে চান
```

---

# 28. Final mental model

CPU এবং GPU concurrent request handling-এর মধ্যে মূল পার্থক্য:

```text
CPU:
Request orchestration machine

GPU:
Parallel math operations execution machine
```

CPU অনেক request-এর state manage করে, I/O wait handle করে, queue চালায়, scheduler চালায়, preprocessing করে।

GPU model computation করে। GPU efficiently ব্যবহার করতে হলে অনেক সময় request-গুলোকে batch করতে হয়।

LLM serving-এ normal batching যথেষ্ট নয়, কারণ response token-by-token generate হয় এবং request length variable হয়। তাই in-flight batching active batch-কে dynamic রাখে:

```text
Finished request বের হয়
New request ঢোকে
KV cache reuse হয়
GPU busy থাকে
Latency কমে
Throughput বাড়ে
```

---

# 29. One-line summary

> CPU concurrent request manage করে scheduling, threading, async I/O এবং queue দিয়ে; GPU concurrent inference serve করে batching, dynamic batching এবং LLM-এর ক্ষেত্রে in-flight batching দিয়ে, যাতে GPU কম idle থাকে এবং বেশি request/token per second serve করা যায়।

---

[1]: https://docs.python.org/3/library/asyncio-eventloop.html "Event loop — Python 3.14.5 documentation"
[2]: https://www.kernel.org/doc/html/v6.10/scheduler/sched-design-CFS.html "CFS Scheduler — The Linux Kernel  documentation"
[3]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/ "CUDA C++ Programming Guide (Legacy) — CUDA C++ Programming Guide"
[4]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html "Dynamic Batching & Concurrent Model Execution — NVIDIA Triton Inference Server"
[5]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_execution.html "Concurrent Model Execution — NVIDIA Triton Inference Server"
[6]: https://nvidia.github.io/TensorRT-LLM/_cpp_gen/executor.html "Executor — TensorRT LLM"
[7]: https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html "Multi-Head, Multi-Query, and Group-Query Attention — TensorRT-LLM"
[8]: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams "Nvidia CUDA Streams"

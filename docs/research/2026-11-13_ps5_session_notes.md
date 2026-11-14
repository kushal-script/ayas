# PS5 organiser session (13 Nov 2026): takeaways

**Source.** The recording `audio.m4a` (23.5 min), transcribed locally with whisper.cpp `small.en`, plus 4 slide photos. The raw transcript is at the end of this file.

**Known mis-hearings in the transcript:** "NAMA" / "LAMBA" = **Llama**; "Halo" = **Hailo**; "execute torch" / "Google torch" / "H.G.ROG" = **ExecuTorch**; "PCV" = **PCB**; "log in the cloud" = **"no cloud"**.

## What the organisers said

### The task

The system must do 5 things:
1. **Detect** surface defects, cracks and anomalies.
2. **Classify** each defect's type and severity.
3. **Alert** the operator with actionable advice.
4. **Log** every inspection as a structured record.
5. **Run at the edge** in real time, with no cloud.

It is **not an image-classification project**. It is a complete system, from the camera to the log.

### Scope

**Pick ONE application area and a SMALL number of defect types.** The listed areas are:
- PCB inspection
- **steel and metal surfaces**
- textile and fabric
- packaging integrity
- automotive components
- pharma tablets and capsules

### Hardware

- **Allowed:** Pi 5 CPU-only, Pi 5 + AI HAT+ / HAT+ 2, other Arm Linux boards (Jetson, Kria), Arm Android phones with a camera.
- **CPU-only is fully allowed and is not scored lower.** What is judged is a correct end-to-end pipeline with good latency and FPS.
- **Memory:** a 4 GB Pi is tight and 8 GB is safer. **Measure memory use early.**

### Reference design

The pipeline has 6 steps:
1. Camera.
2. Pre-processing (resize, clean).
3. Vision model (detect + classify).
4. **Defect record** (type, severity, location, confidence).
5. **Llama via ExecuTorch** turns the record into advice.
6. Operator screen: alert, advice and log.

### Llama

- **Llama gets STRUCTURED findings, not images.** "The vision model finds the defect; Llama explains it and suggests the next action." Llama must never be asked to find defects in pictures.
- **Mandatory:** a Llama-family model (3.2 1B or 3B Instruct) does the advice part, deployed on the device with ExecuTorch.
- **Optional:** everything else (Hailo, YOLO, OpenCV, anomaly detection).
- **Not required:** fine-tuning. Good prompts with structured inputs are enough.
- Llama must be **a real part of the system, "not a decoration"**.
- **Hailo:** if it is used, vision runs on Hailo through HailoRT and Llama on the CPU through ExecuTorch, as **separate pipelines explained in the report**. The Hailo compiler needs x86 Linux (a VM is fine). Running Llama on the HAT+ 2 is only an extra; the submission must use ExecuTorch.

### Data and severity

- **Datasets:** the suggested ones are only suggestions. Public datasets and your own images are both fine.
- **Severity has no standard scale.** Define it for your industry from **size, location and impact**. **Model confidence is NOT severity.**
- **Pass/fail can be rule-based.** Llama explains and suggests the action.

### Demo and offline

- **No conveyor is needed.** A fixed jig with a camera trigger, a live camera, or even a recorded video is fine.
- **Show that the full system works without the internet.** Inference must be strictly edge-based.

### Performance

- **No FPS target.** Report FPS, per-image latency and camera-to-result time, and balance them against accuracy.
- **Benchmarks to report:**

  | Area | Metrics |
  |---|---|
  | Vision | precision, recall, F1 |
  | System | FPS, memory, power |
  | Llama | time to first token, tokens/s, total response time |

- **If stuck at 3–4 FPS:** time pre-processing, inference and post-processing separately, and time vision separately from Llama. Then try a smaller input size, quantisation or a lighter model.
- **If subtle defects are missed:** try higher resolution, better lighting, image tiling and multi-scale detection, and track recall on the hard cases.
- **If modules fail when joined:** agree **one JSON hand-off (defect, severity, confidence, box)**, test each interface alone, and integrate early.
- **Llama tokens/s:** no target; measure it on your own Pi.

### Dates

| Date | Milestone |
|---|---|
| **23 Nov 2026** | Phase 1 submission |
| Mid-Dec 2026 | Shortlist |
| 5 Feb 2027 | Phase 2 proof of concept |
| 15–16 Feb 2027 | Finals: live demo + Q&A |

### Suggested to have by 23 Nov (slide)

1. Chosen application, dataset and defect classes.
2. A working detector with precision and recall.
3. Llama running via ExecuTorch on the device.
4. Vision → Llama hand-off and a local log.
5. An offline run with FPS, latency and memory numbers.
6. GitHub repo, demo video and a report draft.

### Final advice

**"Build the simple end-to-end version first** (camera → model → record → Llama → log). A simple pipeline running on the device is much better than a great model. Don't worry about accuracy first."

**Contact:** tech-queries@armbharatchallenge.com

## What this means for TriNetra (decided 14 Nov)

- **Application:** steel and metal surfaces. The user's added requirement is that it **works irrespective of the lighting conditions**.
- **Priority order:**
  1. The end-to-end pipeline on the Pi.
  2. Llama via ExecuTorch.
  3. Benchmarks.
  4. Detector quality.
  5. The digital twin, used only where it helps lighting robustness.

## Raw transcript (whisper.cpp small.en, unedited)

And all of this must run on the device itself and log in the cloud.
So your system should do five things. One, detect the surface defects, cracks or other problems.
Two, classify the defect, meaning its type and its severity. Three, allow the operator with clear
advice on what to do. Four, keep a record of every inspection. And five, do all this in real
time at the end without depending on the cloud. So when we say edge, we just say that the AI
problems are the small device near the camera. Please remember one thing here. This is not an
image classification project. It is a complete system end to end from the camera to the log.
So you can choose any application area you like. For example, PCV inspection,
steel and metal surfaces, textile and fabric packaging, automotive parts or pharma tablets
and capsules. My advice here is to pick one area and a small number of defect types.
Please do not try to build a system for everything. For hardware, the efficient options are these
Raspberry Pi 5 with CPU only, Raspberry Pi 5 with the AI hat plus or AI hat plus 2,
other ARM based Linux devices or ARM based Android device with the camera. So you do not need an
accelerator. A CPU only 5, Raspberry Pi 5 is fully allowed. We also mentioned within the challenge
page. Most of the questions were based on these only, but the Raspberry Pi 5 with the CPU would be
just sufficient for this project. So here is a simple design with six steps. Step one is the
camera takes the image. Step two will be precast a step. That means resizing and cleaning the image.
You should be done with the problem to be right. And step three, the provision model detects,
classifies and reports the defect. Step four, we make a defect record with the type severity,
location and confidence. And step five, this record goes to NAMA running with a Google torch,
which gives the advice. And step six, the operator's screen shows the alert,
the advice and the love. So the most important point I have mentioned here at the end of this
slide, give NAMA structure findings that you found from the computer vision model,
but okay. And not just draw images, okay. Because the vision model finds the defect,
NAMA explains it and such as the next action. So please will not expect NAMA to find
defects from the pictures alone. And if you use the Halo accelerator,
you will have to separate pipelines. The vision model runs on the Halo chip through Halo RT.
NAMA runs on the Raspberry Pi CPU to execute torch. They are separate and you should explain
more clearly in the report. If you do not have Halo, that is also fine. The vision model can run
off the Pi CPU and both those are valid. And now the technology standard. We only
require technology for mandatory one because we function on the challenge page. It's the
normal. The rule says that you must use some logic from the NAMA family. For example, NAMA
is going to one billion or three billion instruct for the advice part. And you must deploy it on
the device using execute torch. Everything else is optional. And that includes opens in the Halo SDK,
Halo, Retribution Type for detection in the normally detection, which is useful when you have very good examples.
So you are free to choose any computer vision model or any input by the mandate.
It's basically your NAMA code that you launched. And so the data sets, the suggested data sets,
we have mentioned here. So these are only suggestions. You can also collect your own images
and read them and use them for your detection. And so if execute torch is new, there is a free online
course on the online site called Optimize and Deploy AI Inference with Execute Torch on ARM.
So I will give the link to you here in the chat box. I'd love to do a session.
And for submission, you have, so for the submission, you have for deliverables,
for source code, the working application, a demo video, and a code report. And the project has
been hosted on GitHub and submitted to the challenge path. At the bottom of the slide,
you can see a suggested evaluation plans. It covers detection comedy, real-time performance
on the device, useful advice, operator experience, and the report. Please note that this is our
guide to what code looks like. It is my application markdown state. And the phase one deadline is
23rd November, 2026. And so this is the discussion on the problem statement. And let us move on to
the last questions. These are the questions we will see most often. And then go through them
quickly. If something was not clear, please ask at the end of the session. Also, is LAMBA with
execute, first question, is LAMBA with execute torch mandatory? Yes, this is one hard requirement.
LAMBA family model must do the advice part and it must run on the device with execute torch. This is
mandatory. And question number two, does the vision model also mean execute torch? For vision, you can
use x-ray time, TFLite, or anything. It is your choice. Only LAMBA practice type to execute torch.
Are halo, YOLO, and OpenCV compulsory? No, all of them are optional.
I mean, if you're using the CPU, only last one five, it is an efficient factor.
Question number four, do we have to bind to LAMBA? No, you don't start with LAMBA.
LAMBA is going to 1 billion or 3 billion in such value. And write good terms with
structured inputs. Time to only labor if you have time and free of use. And question number five,
can any imprints run in the cloud? No, imprints are strictly edge based. You cannot run
anything on the cloud online. Everything should be offline at the edge. And even in your demo,
please show that the full system works with the internet. Because we want to
see where there was a total input pipeline is running the pipeline.
Question number six, can LAMBA find the defects from the image? No, the vision model defects and
classifies. LAMBA explains the results using the structured findings. Question number one,
which is also a very frequently asked one is whether Raspberry Pi 5 should be used
with a 4GB variant or a 8GB variant. So 4GB is not restricted, but it is very tight
for the environment. We took the LAMBA provision model and your app in the memory of it.
So 4GB would be a little too tight for this, and 2GB safer. Whatever you choose, please,
please measure your memory use early, so that you cannot get restricted by it at the end of the
challenge. Question number two is CPU only is brought equally to the AI hardware solution.
So CPU only is an operation platform, but we cannot. I mean, definitely your project would not be
evaluated based on whether you are using the AI hardware CPU only. You should be really worried
about whether your end-to-end pipeline is working correctly, and with a good latency and with a good
FPS and everything. So this requirement doesn't actually matter much.
Question number three, when hardware or cloud GPUs will be provided?
So we do not provide any cloud GPUs to you. You can use the platforms that travel or
Google will have to run your models and GPUs. I mean, there are many new examples of websites
and platforms where you can use GPUs for free. So you can explore them. We have
specifically providing GPUs for you during your models.
Question number four, can you use Jetson, Kria, or other clouds? So ARM-based Linux devices are
allowed, and Jetson, Kria, and Google have ARM processors. So these are allowed. So everything
that has ARM processors is allowed, and this was not to mention in the challenge page.
So yeah, these are allowed. But you also have to make sure that your software stack and everything
is running on these boards correctly with a good RAM and everything measured correctly.
So make sure that you are keeping in mind all of this. Question number five, how do we compare
with Halo without an x86 Linux machine? So the Halo compiler generally needs an x86 Linux machine.
You can use a virtual machine rather than a Linux machine. The Pi only runs the compiled model,
but they run it. So please, I'll check with the compiler and run my version of machine.
Question number six, AI hat plus two can run LLMs. Can we use it for llama instead?
So AI hat plus two can run language models, which is models, but the rule asks for llama
with its products. So please save it. I mean, llama execute it for your submission. You can try the
Halo LLM option as an extra for the model. Moving on to the data questions that we get very often.
Question number one is, do we need a conveyor code for the demonstration of the project?
So no, we do not need any conveyor code. A fixed chip with a camera trigger and a live
camera or even a recorder, which was fine. What matters is that you show a working inspection
loop end to end, from the detection to the labs and advising. Question number two, which data set
should we use? So no, I mean, we do not provide any data set. We have suggested some. You are
free to choose any other data set you find online, like some publicly available data set. So again,
your own data set, that is also fine. You can make a data set, I mean, the data set by first
collecting the images and then annotating them and everything. That would be a tedious process,
but still you are free to do that. But we have just mentioned some data sets as sufficient only.
So now you are not restricted to using any of these data sets. Those are just suggestions.
Question number three, how do we define severity? So there is no standard scale on how you define
severity. You define it for your application, for the industry that you're choosing. You can use
the medium high such a general transfer classifying the severity. And based on the
different type that you find the size, location and impact the product.
And remember, the confidence over of the model is also not severity. It does not come under severity.
It only tells you how probable it is that your defect is found. So that does not come under
severity. So that shouldn't be a measure for you to classify the severity.
Question number four, can pass or fail the road based? So yes, on the family road based,
you find some simple roads. And even in the industry level, we have a specific set of roads
to define whether the disease is a defect or not. So you can define some roads and decide whether
it's a pass or a fail. And then LAMA explains the defect and suggests the action. But LAMA should
still be a real part of the system. It should just be a decoration. It should be a real part
of the system. And when we have the LAMA is at the end and it should be a part of the system.
So question number six, how do we choose the scope of the project?
So first, start with picking one industry, one inspection problem and a few defect classes.
For example, each of the inspection will just
broken tracks and coverages is good for this type.
Coming to the performance questions. Question number one, how is real time judged?
So FPS number has been given on it. So you are not restricted to having 60 FPS.
So please look out your FPS, the time for image and the time from camera to result.
Make sure that you have a good balance and the FPS, the latency and also the overall
time that you are taking for the ongoing pipeline and also the accuracy.
So you should have a good balance among all of these and back with your great project.
And question number two, what should we benchmark? Okay, so these are the metrics that you have to
report to us. So for the vision model, report position, recall and effort. For the system,
FPS, memory and power. And for LAMA, time to first open,
to things for seconds and to open this one's time. So our views measure the real device.
And question number three, we are stuck at 3 to 4 FPS, what can we do?
So first measure each step separately with lot of the processing,
modeling for end and post-processing, then try a smaller input size,
quantization or a lighter model. So time the vision part separately from LAMA.
So this will give you a timely investigation on all of the parts. And then you can see
why you are getting such a low FPS and then you'll debuff it from there.
Question number four, our model uses subtle defects. Okay, so the model that you are trained
is basically missing some of the smaller defects. Going back, you can try a higher resolution
of the image, better lighting in the image, image painting detection, all of these are
computer vision techniques. You can use them. And check your recall, the hard cases.
Question number five, our models work alone, but fail when joined.
Okay, take one simple handoff format, okay, such as now Jason would depict
severity components in the box, and then test each connection alone. And connect the vision
to LAMA early so that you will have an idea on what all problems you are getting.
Because if you take it early, you will have time to work on it further.
Question number six, is there a 10s per second target for LAMA on the fly?
So no, I mean, we did not get 10s per second target in our challenge description.
But it depends on your quantization, your context language setup. So just measure it on your own
by and in your final report. So let me finish the effect use of birth dates.
Phase one is due on 24th of October 2006. Shortlisted teams will be announced in mid-December.
Phase two, the proof of concept is due on February 27. The findings are on 15th and 16th
of February 27, with a live virtual timer and a journey question and answer session.
And for phase one, here is what I suggest you have ready. This is my suggestion,
not an operation don't work. One, you must choose your application data set and defect classes.
Two, problem detector with precision and recall numbers. And three, LAMA running
with H.G.ROG on your device. And for being in hand of vision to LAMA, you can look at that.
Another important thing is an offline run with FPS latency and memory numbers reported.
And you should have your kit or record. So in short, you must have your at least your
initial submission must have an end-to-end pipeline. But the first thing is the initial execution.
You can improve on the numbers, the latency, and in your trace to timeline.
So if you have any more questions, technical questions that you get, you can always reach
out to us at tech queries at the rate of challenge.com. And you can ask your questions there.
So my final advice is that with the simple end-to-end version first,
camera, which is not going to affect record, LAMA, for example, runs backwards, you can improve on
it after the end-to-end display. So a simple test done by if you are doing runs on the device
is much better than a great model. So just not worry about the accuracy first, just
work on the end-to-end pipeline, then you can improve on the numbers subsequently in the
next phase. So that completes the FAQs. Now the code is open, you can ask more questions.
[BLANK_AUDIO]

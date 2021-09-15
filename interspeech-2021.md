## Interspeech 2021

### General thoughts
- There's a lot of effort in making STT really work in production, implying this is by far not yet solved. In production systems, you need to optimize for resources (cpu, mem), latency, accuracy of rare words (long tail), efficiently incorporate contextual information, etc.
- On-device STT is a big topic, likely due to huge potential in cost savings given the increasing usage, better latency and privacy.
- RNN-T is still the most common architecture in streaming STT.
- Still missing a good public STT dataset with a realistic production-like scenario at a reasonable scale. Top players have in-house datasets of tens of thousands transcibed hours from production. Public datasets are either scripted or small (or both). One of the papers even shows the most common public datasets generalize very poorly.

### Common techniques
- Transfer learning. Base model (different domain or even language) is fine-tuned (part of it or full) for target data.
- Unsupervised learning. The data is unlabeled (audio only). Variants of wav2vec.
- Self-supervised learning. Most of your data is unlabeled. Iterative training using labels from previous model iteration.
- Training on TTS-generated data. Since the recent TTS models are mostly generating mels or similar features, no need to actually generate wav, train STT directly on mels. A lot of research around how to do this properly, e.g. which parts of the models to freeze, introduce context vectors for training on TTS vs real data, improve TTS variability, etc.
- Shallow fusion. Combines model outputs with external LM outputs.

### Selected papers

#### Rethinking Evaluation in ASR: Are Our Models Robust Enough? (Facebook)

Points out huge differences between datasets and the inability to generalize, e.g. a model trained on Common Voice performs poorly on Switchboard and vice versa.

#### An Efficient Streaming Non-Recurrent On-Device End-to-End Model with Improvements to Rare-Word Modeling (Google)

Google keeps improving its on-device STT, optimizing for latency, footprint and accuracy of the long tail of rare words.
They use wordpiece RNN-T model with a bunch of improvements coming from recent papers.

- Conformer layers (self-attention and convolution) instead of LSTM in encoder.
- Cascaded encoders - first pass using causal encoder (no lookahead), second pass adds a few layers on top of the previous encoder layers and take left and right context.
- FastEmit - improves latency by encouraging the model to output non-blank symbols.
- Prefetching - since there's typically a downstream task (like search), the task is performed using partial results to improve overall latency in case the final result matches the partial result.
- HAT (Hybrid Autoregressive Transducer) factorization - a way to improve shallow fusion with external language model by factoring out internal RNN-T LM score.
- Neural LM - Conformer LM to rescore hypotheses from the second pass. You wouldn't use such large LM in shallow fusion, because it's expensive.
- RNN-T prediction network (LSTM) replaced by a lookup table.
- 16 dimensional one-hot domain-id vector appended to the standard logmel feature frontend.

#### Noisy student-teacher training for robust keyword spotting (Google)

Keyword spotting in low-resource scenario is nowadays done using an LSTM, which outputs per-frame results (K classes for K possible keywords), smoothing (average over a sliding window) and a threshold. Then you can balance false accepts and false rejects by changing the threshold.

This paper shows noisy teacher-student approach, where the model is trained iteratively using unlabeled data and soft-labels given the previous iteration of the model. Moreover the data is highly augmented using SpecAugment. Aggressive SpecAugment is shown to degrade performance in supervised setting (hard labels), but
improves while using soft-labels. The intuition is that soft-labels adapt to the degree of degradation of the input audio.

#### Low Resource ASR: The surprising effectiveness of High Resource Transliteration (IBM)

Transfer learning from high to low resource languages with different character sets. A standard technique called transliteration is used to rewrite transcripts from English to a target language, keeping phonetic similarity. This allows to transfer both encoder and decoder layers and improves over just transferring the encoder.

#### Multiple Softmax Architecture for Streaming Multilingual End-to-End ASR Systems (Microsoft)

Multilingual streaming RNN-T STT done using separate joint networks, softmax layers and embeddings. Encoder and prediction network are shared. In parallel, language identification model using inputs from encoder is run, while also each language has it's own beam search. Finally, the result for the language predicted by LID is returned.

This is shown to be better than a single softmax with unified token set and also it's easier to extend by a new language.

#### Fast Text-Only Domain Adaptation of RNN-Transducer Prediction Network (Speechly)

A very interesting and simple alternative to shallow fusion. Since prediction network cannot be trained directly using text-only data,
they add an LM component on top of it, train it using train transcriptions while freezing prediction network and then train prediction network
using the domain text data while freezing the LM component. This way, there's no change in inference, you just get an adapted prediction network.

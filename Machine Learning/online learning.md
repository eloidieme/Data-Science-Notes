You’re continuously updating your models with new batches of data. You’re not re-training the model as you would in offline learning, you’re simply updating the model weights based on new observations.

In online learning, you train the system incrementally by feeding it data instances sequentially, either individually or in small groups called mini-batches. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives

![[online_learning.png]]

Online learning algorithms can also be used to train systems on huge datasets that cannot fit in one machine’s main memory (this is called **_out-of-core learning_**). The algorithm loads part of the data, runs a training step on that data, and repeats the process until it has run on all of the data.

![[out_of_core_learning.png]]

One important parameter of online learning systems is how fast they should adapt to changing data: this is called the **_learning rate_**.

**High learning rate** — Your system will **rapidly adapt to new data**, but it will also tend to **quickly forget the old data** (you don’t want a spam filter to flag only the latest kinds of spam it was shown).

**Low learning rate** — The system will **learn more slowly**, but it will also be **less sensitive to noise** in the new data or to sequences of nonrepresentative data points (_outliers_).

**A big challenge with online learning is that if bad data is fed to the system, the system’s performance will gradually decline**.

To reduce this risk, you need to monitor your system closely and promptly switch learning off (and possibly revert to a previously working state) if you detect a drop in performance.

You may also want to monitor the input data and react to abnormal data (e.g., using an **_anomaly detection algorithm_**).
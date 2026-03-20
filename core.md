So let’s take the auto research idea from AutoResearch project (see the autoresearch folder under ai) and apply it to training classic machine learning models. In this case, we’ll focus on scikit-learn, which is a Python machine learning library where you can load all kinds of models like regression, logistic regression, support vector machines, and so on.

The problem we want to solve: one of the problems with current AI systems is that we need a simple guardrail layer that can avoid different kinds of attacks from users or other external inputs. Prompt attacks are a big one, and they’re becoming an increasingly serious problem. If you have agents reading emails, online information, or replies from other people, they can easily run into prompt injections that cause issues.

So the idea is: can we train a very simple classifier to identify those problems? Because it would be a simple non-transformer model, it should be more predictable, and because it is simple, it should also have very low latency.

I’ll provide the dataset from Hugging Face. The real goal is to try different ideas on the same dataset, tune the parameters, and improve accuracy, especially validation accuracy. We need to do this properly with a train, validation, and test split. That’s the whole thing.

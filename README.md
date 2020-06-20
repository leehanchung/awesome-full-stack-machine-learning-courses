# Guides and Resources Full Stack Machine Learning Engineering
This is curated guide of resources and case studies for full stack machine learning engineering, break down by topics and specializations. Python is the preferred language of choice as it covers end-to-end machine learning engineering.

The curated resources will be focusing around the infamous hidden technical debt of machine learning paper by Google. Note that machine learning models, while crucial, requires a lot of engineering services and effort to be productized to generate business values. For those who does not have a technical background in or wants some refreshers of computer science, please visit the [computer science section](#Computer-Science).

![Hidden Debt of Machine Learning](docs/assets/technical_debt_ml.png)

# Data Engineering

[SQL for Data Analysis](https://classroom.udacity.com/courses/ud198)

[Spark](https://classroom.udacity.com/courses/ud2002)

#### Data Engineering Frameworks

[Spark](https://spark.apache.org/)

[Airflow](https://airflow.apache.org/)

[dagster](https://docs.dagster.io/)


# Machine Learning Model Serving

If a model was trained on a computer and no API is around to serve it, can it make an inference?

#### :school: Courses
[Berkeley: Full Stack Deep Learning](https://fullstackdeeplearning.com/) :star:

[Udemy: Deployment of Machine Learning Models](https://www.udemy.com/course/deployment-of-machine-learning-models) :star:

[Udemy: The Complete Hands On Course To Master Apache Airflow](https://www.udemy.com/course/the-complete-hands-on-course-to-master-apache-airflow)

[Pipeline.ai: Hands-on with KubeFlow + Keras/TensorFlow 2.0 + TF Extended (TFX) + Kubernetes + PyTorch + XGBoost](https://www.youtube.com/watch?v=AaBqhGEwxXI)

#### Model Serving Frameworks

[Google: Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving)

[Sheldon](https://www.seldon.io/)

[Cortex](https://docs.cortex.dev/)

[Google: KF Serving](https://www.kubeflow.org/docs/components/serving/)

[Flask](https://flask.palletsprojects.com/en/1.1.x/api/)

[FastAPI](https://fastapi.tiangolo.com/)

[RedisAI](https://oss.redislabs.com/redisai/)

[Lyft: FlyteHub](https://flytehub.org/) [example](https://flytehub.org/objectdetector)

[Uber: Neuropod](https://eng.uber.com/introducing-neuropod/)


# Machine Learning Operations (MLOps)

## Case Studies and Tutorials
[Tutorial: From Notebook to Kubeflow Pipeline](https://www.youtube.com/watch?v=C9rJzTzVzvQ)

[How to version control your production machine learning models](https://algorithmia.com/blog/how-to-version-control-your-production-machine-learning-models)

[Microsoft Azure ML Ops Python](https://github.com/microsoft/MLOpsPython)

[Production Data Science](https://github.com/FilippoBovo/production-data-science)

## Continous Integration/Continous Delivery

[GCP ML Pipeline Generator](https://github.com/GoogleCloudPlatform/ml-pipeline-generator-python)

[Github Actions ML Ops](https://mlops-github.com/)

[Github Actions ML Ops abuse](https://docs.google.com/presentation/d/1aIwxTMPF8rm2sY3VAJypB5x7paYeCVMlpwYsSHrUl_E/edit) [repo](https://github.com/peckjon/coreml_ghactions)

## Pipeline Tools

[Kubeflow](https://www.kubeflow.org/)

[MLflow](https://mlflow.org/)

[Allegro.ai](https://allegro.ai/)

[Cnvrg.io](https://cnvrg.io/)

[mleap](https://mleap-docs.combust.ml/)

[mlrun](https://www.iguazio.com/open-source/mlrun/)


# Machine Learning Project Design Case Studies

#### :newspaper: Articles
[Microsoft: The Team Data Science Process lifecycle](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/lifecycle)

[Microsoft: Software Engineering for Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2019/03/amershi-icse-2019_Software_Engineering_for_Machine_Learning.pdf)


[Google: Machine Learning: The High Interest Credit Card of Technical Debt](https://ai.google/research/pubs/pub43146)

[Amazon: Introducing the Well Architected Framework for Machine Learning](https://aws.amazon.com/blogs/architecture/introducing-the-well-architected-framework-for-machine-learning/)

[How do Data Science Workers Collaborate? Roles, Workflows, and Tools](https://arxiv.org/abs/2001.06684)

[Software Engineering for Machine Learning: A Case Study](https://ieeexplore.ieee.org/document/8804457)

[Case Studies](MLE_Case_Studies.md)

[Spotify: The Winding Road to Better Machine Learning Infrastructure Through Tensorflow Extended and Kubeflow](https://labs.spotify.com/2019/12/13/the-winding-road-to-better-machine-learning-infrastructure-through-tensorflow-extended-and-kubeflow/)

[Toutiao (ByteDance/Tik-Tok): Recommendation System Design](https://leehanchung.github.io/2020-02-18-Tik-Tok-Algorithm/)

[DailyMotion: Industrializing Machine Learning Pipelines](https://fr.slideshare.net/GermainTanguy/industrializing-machine-learning-pipelines)

[Paypal: On a Deep Journey towards Five Nines](https://www.infoq.com/presentations/journey-99999/)


# Machine Learning Modeling

Fundamentals of machine learning, including linear algebra, vector calculus, and statistics.

#### :books: Textbooks
[Mathematics for Machine Learning](https://mml-book.github.io/)

[Concise Machine Learning](https://people.eecs.berkeley.edu/~jrs/papers/machlearn.pdf)

[The Elements of Statistical Learning](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)

[Mining of Massive Datasets](http://www.mmds.org/)

[Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf): [[Codes](https://github.com/ctgk/PRML)]

#### :school: Courses
[MIT 18.05: Introduction to Probability and Statistics](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/) :star:

[MIT 18.06: Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/) :star:

[Stanford Stats216: Statiscal Learning](https://lagunita.stanford.edu/courses/HumanitiesSciences/StatLearning/Winter2016/about) :star:

[CalTech: Learning From Data](https://work.caltech.edu/telecourse.html)

[edX ColumbiaX: Machine Learning](https://www.edx.org/course/machine-learning)

[Stanford CS229: Machine Learning](https://see.stanford.edu/Course/CS229)

[Stanford CS246: Mining Massive Data Sets](http://web.stanford.edu/class/cs246/)

[Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)

## Artificial Intelligence

Machine learning is a sub field of Artificial Intelligence. These courses provides a much higher level understanding of the field of AI.


#### :books: Textbooks

[Artificial Intelligence: A Modern Approach](https://www.amazon.com/Artificial-Intelligence-Modern-Approach-3rd/dp/0136042597)

#### :school: Courses

[Berkeley CS188: Artificial Intelligence](https://edge.edx.org/courses/course-v1:BerkeleyX+CS188+2018_SP/course/) :star:

[edX ColumbiaX: Artificial Intelligence](https://www.edx.org/course/artificial-intelligence-ai): [[Reference Solutions](https://github.com/leehanchung/CSMM-101x-AI)]

## Deep Learning Overview

Basic overview for deep learning.

#### :school: Courses
[Deeplearning.ai Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning): [[Reference Solutions](https://github.com/leehanchung/deeplearning.ai)] :star:

[Fast.ai Part 2](https://course.fast.ai/part2)


## Specializations

### Recommendation Systems

### Vision

#### :books: Textbooks
[Deep Learning](http://www.deeplearningbook.org/)

#### :school: Courses
[Stanford CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/): [[Assignment 2 Solution](https://github.com/leehanchung/cs182/tree/master/assignment1), [Assignment 3 Solution](https://github.com/leehanchung/cs182/tree/master/assignment2)] :star:

[Berkeley CS182: Designing, Visualizing, and Understanding Deep Neural Networks](https://bcourses.berkeley.edu/courses/1478831/pages/cs182-slash-282a-designing-visualizing-and-understanding-deep-neural-networks-spring-2019): [[Reference Solutions](https://github.com/leehanchung/cs182)]


### Natural Language Processing

With languages models and sequential models, everyone can write like GPT-2.

#### :books: Textbook
[Deep Learning](http://www.deeplearningbook.org/)

[Introduction to Natural Language Processing](https://www.amazon.com/Introduction-Language-Processing-Adaptive-Computation/dp/0262042843)

#### :school: Courses
[Stanford CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/): [[Reference Solutions](https://github.com/leehanchung/cs224n)] :star:

[Berkeley CS182: Designing, Visualizing, and Understanding Deep Neural Networks](https://bcourses.berkeley.edu/courses/1478831/pages/cs182-slash-282a-designing-visualizing-and-understanding-deep-neural-networks-spring-2019): [[Reference Solutions](https://github.com/leehanchung/cs182)]


### Deep Reinforcement Learning

#### :books: Textbook

[Reinforcement Learning](http://www.incompleteideas.net/book/the-book.html)

[Deep Learning](http://www.deeplearningbook.org/)

#### :school: Courses
[Coursera: Reinforcement Learning Specialization](https://www.coursera.org/specializations/reinforcement-learning) <= Recommended by [Richard Sutton](https://www.reddit.com/r/MachineLearning/comments/h940xb/what_is_the_best_way_to_learn_about_reinforcement/), the author of the de facto textbook on RL. :star:

[Berkeley CS182: Designing, Visualizing, and Understanding Deep Neural Networks](https://bcourses.berkeley.edu/courses/1478831/pages/cs182-slash-282a-designing-visualizing-and-understanding-deep-neural-networks-spring-2019): [[Reference Solutions](https://github.com/leehanchung/cs182)]

[Stanford CS234: Reinforcement Learning](https://web.stanford.edu/class/cs234/)

[Berkeley CS285: Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/) :star:

[CS 330: Deep Multi-Task and Meta Learning](http://cs330.stanford.edu/): [Videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMC6zfYmnD7UG3LVvwaITY5)

[Berekley: Deep Reinforcement Learning Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)

[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)


### Unsupervised Learning and Generative Models

#### :school: Courses
[Stanford CS236: Deep Generative Models](https://deepgenerativemodels.github.io/)

[Berkeley CS294-158: Deep Unsupervised Learning](https://sites.google.com/view/berkeley-cs294-158-sp19/home)


### Robotics :robot:

#### :school: Courses
[ColumbiaX: CSMM.103x Robotics](https://courses.edx.org/courses/course-v1:ColumbiaX+CSMM.103x+1T2020/)

[CS 287: Advanced Robotics](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/)


# Computer Science

Basic computer science skill is required for machine learning engineering.

### :books: Textbooks
[Grokking Algorithms](https://github.com/KevinOfNeu/ebooks/blob/master/Grokking%20Algorithms.pdf)

[Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)

### :school: Courses
[MIT: The Missing Sememster of Your CS Education](https://missing.csail.mit.edu/) :star:

[Corey Schafer Python Tutorials](https://www.youtube.com/watch?v=YYXdXT2l-Gg&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU)

[edX MITX: Introduction to Computer Science and Programming Using Python](https://www.edx.org/course/6-00-1x-introduction-to-computer-science-and-programming-using-python-4) :star:

[edX Harvard: CS50x: Introduction to Computer Science](https://www.edx.org/course/cs50s-introduction-to-computer-science) 


# LICENSE
All books, blogs, and courses are owned by their respective authors.

You can use my compilation and my reference solutions under the open CC BY-SA 3.0 license and cite it as:
```
@misc{leehanchung,
  author = {Lee, Hanchung},
  title = {Full Stack Machine Learning Engineering Courses},
  year = {2020},
  howpublished = {Github Repo},
  url = {https://github.com/full_stack_machine_learning_engineering_courses}
}
```

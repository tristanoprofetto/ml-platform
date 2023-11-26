import pandas as pd

reflections = [
    "I enjoyed learning about numbers and how to add them. I also liked drawing pictures of my family.",
    "I liked learning about the different types of animals. I also liked learning about the different types of plants.",
    "I found fractions challenging but interesting. We also started a project on ancient Egypt, which is exciting!",
    "We worked on multiplication and division. I also enjoyed our group project on renewable energy sources.",
    "This week we focused on grammar and writing essays. I'm also working on a science project about the solar system.",
    "We discussed more complex math problems and started learning about the American Revolution in history class.",
    "I had trouble understanding addition and felt a bit lost during math class. I also struggled with writing essays.",
    "It was hard to keep up with the reading about animals. I didn't find the story interesting.",
    "Multiplication and division are challenging, and I'm not liking our group project much.",
    "Fractions are really confusing for me, and I'm not enjoying the ancient Egypt project.",
    "I'm struggling with grammar and essay writing. The science project is overwhelming.",
    "I'm having a hard time with math and history. I'm not enjoying the science project.",


]

sentiment =[
    "positive",
    "positive",
    "neutral",
    "neutral",
    "neural",
    "neutral",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
]

print(len(reflections) == len(sentiment))

df = pd.read_csv('./data/feedback.csv')

print(df.groupby('label').count())
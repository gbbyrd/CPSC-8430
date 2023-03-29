from transformers import pipeline

model_checkpoint = 'checkpoints'
question_answerer = pipeline("question-answering", model=model_checkpoint)

context = """The 1972 season proved to be a breakout year for the Tigers. The Tigers 
went undefeated in conference play to capture the first of eight straight ACC titles 
and finished the year with a 13–1–1 record and earned their first trip to the NCAA tournament. 
The 1973 season would prove to be even more successful, as the Tigers went 16–1 and made it to 
the semifinals of the NCAA tournament. By the end of the decade, the Tigers had 8 conference 
titles, 3 trips to the round of 16 in the NCAA tournament, an Elite 8 appearance, 3 Final Four 
appearances, and finished the 1979 season as national runners-up."""

while 1:

    question = input("Please input a question:")

    answer = question_answerer(question=question, context=context)

    print(answer)
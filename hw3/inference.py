from transformers import pipeline

model_checkpoint_spoken = 'checkpoints/spoken_squad'
model_checkpoint = 'checkpoints/squad'
question_answerer_spoken = pipeline("question-answering", model=model_checkpoint_spoken)
question_answerer = pipeline("question-answering", model=model_checkpoint)

context_spoken = """super bowl fifty was an american football game to determine the 
champion of the national football league nfl for the twenty fifteen season. the 
american football conference a f c c champion denver broncos defeated the national 
football conference n f c c champion carolina panthers twenty four to ten to earn 
their third super bowl title. the game was played on february seventh twenty sixteen 
and levis stadium in the san francisco bay area santa clara california. as this was 
the fiftieth super bowl the league emphasized the golden anniversary with various 
goldsteins initiatives as well as temporarily suspending the tradition of naming 
each super bowl game with roman numerals under which they gain would have been known as 
super bowl l sell that the logo could prominently featured the arabic numerals fifty."""

context = """Super Bowl 50 was an American football game to determine the champion 
of the National Football League (NFL) for the 2015 season. The American Football 
Conference (AFC) champion Denver Broncos defeated the National Football Conference 
(NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The 
game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay 
Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized 
the "golden anniversary" with various gold-themed initiatives, as well as temporarily 
suspending the tradition of naming each Super Bowl game with Roman numerals (under 
which the game would have been known as "Super Bowl L"), so that the logo could 
prominently feature the Arabic numerals 50."""

while 1:

    question = input("Please input a question:")

    answer_spoken = question_answerer_spoken(question=question, context=context_spoken)
    answer = question_answerer(question=question, context=context_spoken)
    print(f'spoken answer: {answer_spoken}')
    print(f'answer: {answer}')
    
    answer_spoken = question_answerer_spoken(question=question, context=context)
    answer = question_answerer(question=question, context=context)
    print(f'spoken answer: {answer_spoken}')
    print(f'answer: {answer}')
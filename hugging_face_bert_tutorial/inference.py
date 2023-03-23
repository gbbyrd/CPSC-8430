from transformers import pipeline

model_checkpoint = 'checkpoints'
question_answerer = pipeline("question-answering", model=model_checkpoint)

context = """versions of the " doctor who theme " have also been released as pop music over the years. in the early 1970s, jon pertwee, who had played the 
third doctor, recorded a version of the doctor who theme with spoken lyrics, titled, " who is the doctor ". [ note 6 ] in 1978 a disco version 
of the theme was released in the uk, denmark and australia by the group mankind, which reached number 24 in the uk charts. in 1988 the band the 
justified ancients of mu mu ( later known as the klf ) released the single " doctorin ' the tardis " under the name the timelords, which reached 
no. 1 in the uk and no. 2 in australia ; this version incorporated several other songs, including " rock and roll part 2 " by gary glitter 
( who recorded vocals for some of the cd - single remix versions of " doctorin ' the tardis " ). others who have covered or reinterpreted the 
theme include orbital, pink floyd, the australian string ensemble fourplay, new zealand punk band blam blam blam, the pogues, thin lizzy, dub 
syndicate, and the comedians bill bailey and mitch benn. both the theme and obsessive fans were satirised on the chaser ' s war on everything. 
the theme tune has also appeared on many compilation cds, and has made its way into mobile - phone ringtones. fans have also produced and 
distributed their own remixes of the theme. in january 2011 the mankind version was released as a digital download on the album gallifrey and 
beyond."""

while 1:

    question = input("Please input a question:")

    answer = question_answerer(question=question, context=context)

    print(answer)

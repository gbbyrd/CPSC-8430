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

context = """Ghast are a large flying mob in the nether that looks like ghosts. Usually when a ghast is flying around, it can be heard making crying noises. But when they notice a player, you’ll see them open their eyes and scream, followed by them shooting a fireball. Even though they’re extremely dangerous, ghast are known to drop valuable loot. Listed below is the amount of damage a ghast will do. Based on the difficulty you’re playing on. Ghast can be found all throughout the Nether, and for them to spawn there needs to be a 5x5 wide block and four blocks high. Listed below are all the biomes you can find a ghast in."""


while 1:

    question = input("Please input a question:")

    # answer_spoken = question_answerer_spoken(question=question, context=context_spoken)
    # answer = question_answerer(question=question, context=context_spoken)
    # print(f'spoken answer: {answer_spoken}')
    # print(f'answer: {answer}')
    
    answer_spoken = question_answerer_spoken(question=question, context=context)
    answer = question_answerer(question=question, context=context)
    print(f'spoken answer: {answer_spoken}')
    print(f'answer: {answer}')
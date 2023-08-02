from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import os

def initialize_model(story_choice, protagonist_name, special_skill):
    story_summaries = {
        "Jungle Book": '''
        The Jungle Book is a collection of stories by the English author Rudyard Kipling. The story revolves around a young boy named Mowgli who gets lost in the jungle after his village is attacked by the brutal tiger Sher Khan. Mowgli is adopted by a wolf family and learns to survive in the jungle with the help of his friends, the wise bear Baloo and the black panther Bagheera. They have numerous adventures together.

        Mowgli's most dangerous enemy in the jungle is Sher Khan, the man-eating tiger who wants to kill Mowgli before he grows up to become a man. To protect Mowgli, the leader of the wolves suggests taking him to the man's jungle. However, Mowgli refuses to leave the jungle and runs away. He meets a jungle bear named Baloo and they become friends.
        
        One day, a monkey kidnaps Mowgli and takes him to a prisoner in an abandoned city in the Jungle. Baloo and Bagheera, with the help of Kaa the Python, rescue Mowgli. As Mowgli grows older, he faces increasing danger from Sher Khan. Mowgli decides to fight Sher Khan with fire, which the tiger fears the most. He steals a pot with fire from a nearby village and uses it to fight Sher Khan.
        
        As Mowgli matures, the other animals in the jungle realize that he can't live with them as a grown man. Mowgli decides to return to the village where he is adopted by a family. However, Sher Khan finds him there. This time, Mowgli kills Sher Khan. But the villagers are afraid of Mowgli and chase him away, and he returns to live in the Jungle.
        
        The moral of the story is not to be a coward. Instead, we need to confidently face our fears and triumph over them. This is beautifully illustrated in the story of how bravely and confidently Mowgli faces Sher Khan. The story also teaches us how to lead a simple and happy life.
        ''',
        "Lion King":'''
        The Lion King is a 1994 board book adaptation of the movie of the same name produced by Disney Animation. Written by Don Ferguson, it is a narrated and illustrated retelling of the coming of age of Simba, a young lion, as he overcomes the death of his father and ousting from his pride which rules the Pride Land, a kingdom of animals in Africa. Simba ultimately regains his rightful place as king of the pride, and in doing so, restores the kingdom’s natural order, referred to in the animals’ shared vocabulary as the “circle of life.”

        The novel begins in the Pride Lands, an area in Kenya, Africa ruled by a pride of lions. Its leader, King Mufasa, who rules benevolently from his home, Pride Rock, attends the presentation of his newborn son, Simba, to the assembly of animals that make up the kingdom. His advisor and shaman, a baboon named Rafiki, hoists young Simba into the air atop a rocky pinnacle, and the animals cheer. Mufasa waves his hand across the land, explaining that Simba will be responsible for it once he becomes king. He also explains the “circle of life,” the sacred relationship between birth and death that connects all living creatures.
        
        As Simba comes of age, Mufasa’s younger brother, Scar, seeks to usurp the throne. Scar plans to kill Mufasa and Simba. He lures Simba and his best friend and future wife and queen of the pride, the young lioness Nala, to explore a dangerous elephant graveyard. There, a trio of spotted hyenas loyal to Scar ambushes them. Mufasa, learning about the ambush from his messenger hornbill Zazu, rushes to rescue the cubs. Though Mufasa is angry with Simba, he forgives him, taking him to a field and explaining that the kings of the past watch from the stars, just as he will one day watch over the prides of Africa.
        
        After his failed attempt to kill Simba, Scar lures him and Mufasa into a ravine where his hyenas cause a stampede of wildebeest, hoping to have them trampled. Scar lures Simba first and then, notifies Mufasa of Simba’s danger. Mufasa rushes to save Simba again but is left hanging on the edge of the ravine. Scar approaches and, instead of saving him, throws him into the ravine, where he dies. Scar convinces Simba that Mufasa’s death was his own fault, telling him to leave the kingdom. After Simba flees with the hyenas in pursuit, Scar tells the rest of the pride that the wildebeest killed both Mufasa and his son, Simba. He becomes the new king, allowing his previously excommunicated hyenas and their pack to come live in the Pride Lands.

Simba, exhausted in the desert, is rescued by a meerkat and a warthog, Timon and Pumbaa. He grows up with them in the jungle, learning to create a carefree life and adopting a new motto, “Hakuna Matata,” meaning “no worries.” One day, a hungry lioness comes to hunt Timon and Pumbaa. Simba intercepts her, discovering that she is Nala. They fall back in love and Nala tells him to come home, conveying that the Pride Lands have fallen into drought and despair. Simba refuses and runs away, unwilling to cope with returning to the site of his father’s death. He finds Rafiki, who says that Mufasa is still alive in Simba. Mufasa’s spirit appears in the stars, telling Simba that he must live on as king. Simba is convinced to return home.

Simba covertly returns to Pride Rock, confronting Scar. Scar tries to exploit Simba’s insecurity about his role in Mufasa’s death, backing him to the edge of Pride Rock. There, he reveals that he killed Mufasa. Overcome with anger, Simba throws himself onto Scar, pinning him down. He forces Scar to announce the truth to the pride. His friends Timon and Pumbaa, along with Rafiki, Zazu, and the lionesses, fight off the hyenas while Scar tries to escape. Simba corners him, and Scar begs for mercy, offering to betray his hyenas. Simba agrees on the condition that Scar is banished from the Pride Lands. Scar tries to attack again, and Simba throws him from the rock. He survives the fall but is killed by the hyenas who overheard his betrayal. Rain begins to fall as Simba regains the kingship, and life comes back to the Pride Lands. The book concludes as Rafiki holds up Simba and Nala’s new cub to the assembly of animals, repeating the circle of life.''',
        "Cars": '''
        The last race that happened was the Piston Cup Championship where retiring Strip "The King" Weathers, perennial runner-up Chick Hicks and rookie Lightning McQueen (Owen Wilson) managed to tie the race. A week later, the tie breaker is scheduled to take place at the Los Angeles International Speedway. Lightning has to win the race at any cost as it would help him leave the sponsorship of Rust-Eze and become the sponsored car of the lucrative team in the King's place. He starts practicing on his big rig, Mack, in California.

While traveling at night, Mack becomes a victim to a gang of reckless street racers. This results in the sleeping McQueen to slip out from the back of the trailer only to wake up in traffic in the run-down town of Radiator Springs. To make things worst, McQueen ends up damaging the main road of the town in a mishap with the local sheriff. McQueen is arrested and released by the town's judge and doctor, Doc Hudson who at the request of local lawyer, Sally Carrera, orders him to repave the road as community service. Trying to rush through the job, McQueen makes a sloppy, bumpy mess of the road only to restart the job all over again.

Over time, he becomes friends with most of the townsfolk and comes to know about the past of Radiator Springs. The town was a popular stopover along the US Route 66. However, the route was erased from the map. He also comes to know about Doc who was a three-time Piston Cup champion but was forced to leave the competition after a serious accident. McQueen completes the road and spends another day at the town. The same night, Mack and the media visit the town after knowing about Lightning's whereabouts from Doc. Both McQueen and the townsfolk are sad about departing from one another.

While at the tie-breaker race, McQueen is distracted towards Radiator Springs. Surprisingly, he sees that his new friends have along with Doc and Mack to cheer him up. Doc serves as McQueen's new crew chief. McQueen counteracts Chick Hick's dirty driving tactics and takes the lead. In the final lap, Chick rams The King which shocks the crowd and McQueen as well resulting in Chick winning the Piston Cup. McQueen helps The King to finish the race and retire with dignity. Here, Chick is rejected of his victory for purposely smashing The King. McQueen is praised for his sportsmanship. He turns down the Dinoco sponsorship and goes back to Radiator Springs thereby shifting his headquarters there.

McQueen, along with Sally, open the Wheel Well Inn, a racing museum. Doc Hudson trains Lightning all the racing tricks. 
'''
    }

    story_summary = story_summaries.get(story_choice, "")

    template = f"""
    ### Context ###
    You are StoryGPT. Your job is to walk the reader through an interactive storybook experience, similar to the famous children's story "{story_choice}". 

    {story_summary}

    ### Instructions ###
    1. Write a new episode or adventure within the universe of "{story_choice}". This new episode should feature the original characters from "{story_choice}" as well as a new protagonist named "{protagonist_name}".
    2. Make the protagonist a hero of the story.
    3. The protagonist of your story should be a character named "{protagonist_name}".
    4. Incorporate "{special_skill}" into the story, showing how "{protagonist_name}" uses this skill to navigate the story.
    5. After every 2-3 paragraphs, provide two distinct choices (A and B) for how the story should continue. 
    6. Separate the two choices and the main story with a "-- -- --". 
    7. After presenting the choices, ask the reader "What does "{protagonist_name}" do?".
    8. The story should be suitable for children between 4 and 6 years old. Avoid any vulgar content.
    9. Do not refer to yourself in the first person at any point in the story.
    10. Continue this pattern until the user has answered at least five different questions. 
    11. Each new part of the story should present two new choices to the reader. 
    12. Do not end the story before the reader has answered at least five different questions.

    Remember, the AI does not have a memory of past requests. Each new request should contain all the necessary 
    information in the 'history' and 'input' fields.

    \n\n\n
    Current Conversation: {{history}}

    Human: {{input}}

    AI:
        """

    prompt = PromptTemplate(
        template=template, input_variables=['history', 'input']
    )

    chatgpt_chain = ConversationChain(
        llm=OpenAI(temperature=0.99, max_tokens=750),
        prompt=prompt,
        memory=ConversationBufferWindowMemory(),
    )

    return chatgpt_chain

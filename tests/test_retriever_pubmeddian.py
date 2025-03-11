"""
 @Author: cks
 @Date: 2025/3/10 11:05
 @Description:

"""
import asyncio
import pprint

from dotenv import load_dotenv

from gpt_researcher.actions.retriever import get_retrievers
from gpt_researcher.config.config import Config

# Load environment variables from .env file
load_dotenv()


async def test_scrape_data_by_query():
    # Initialize the Config object
    config = Config()

    # Retrieve the retrievers based on the current configuration
    retrievers = get_retrievers({}, config)
    print("Retrievers:", retrievers)

    sub_query = "design patterns for autonomous ai agents"

    # Iterate through all retrievers
    for retriever_class in retrievers:
        # Instantiate the retriever with the sub-query
        retriever = retriever_class(sub_query)

        # Perform the search using the current retriever
        search_results = await asyncio.to_thread(
            retriever.search, max_results=3
        )

        print("Search results:")
        pprint.pprint(search_results, indent=4, width=80)

if __name__ == "__main__":
    asyncio.run(test_scrape_data_by_query())

"""
If Run Success, It Will Be Return Like This, and you can limit the data count:

PubmedDianSearch: Searching with query design patterns for autonomous ai agents...
Search results:
[   {   'body': 'Machines powered by artificial intelligence (AI) are '
                'increasingly taking overtasks previously performed by humans '
                'alone. In accomplishing such tasks, they mayintentionally '
                "commit 'AI crimes', ie engage in behaviour which would "
                'beconsidered a crime if it were accomplished by humans. For '
                'instance, an advancedAI trading agent may-despite its '
                "designer's best efforts-autonomously manipulatemarkets while "
                'lacking the properties for being held criminally responsible. '
                'Insuch cases (hard AI crimes) a criminal responsibility gap '
                'emerges since no agent(human or artificial) can be '
                'legitimately punished for this outcome. We aim toshift the '
                "'hard AI crime' discussion from blame to deterrence and "
                "design an 'AIdeterrence paradigm', separate from criminal law "
                'and inspired by the economictheory of crime. The homo '
                'economicus has come to life as a machina economica,which, '
                'even if cannot be meaningfully blamed, can nevertheless be '
                'effectivelydeterred since it internalises criminal sanctions '
                'as costs.',
        'href': 'https://pubmed.ncbi.nlm.nih.gov/39234494/',
        'title': "'Hard AI Crime': The Deterrence Turn."},
    {   'body': 'Norbert Wiener and Nikolai Bernstein set the stage for a '
                'worldwidemultidisciplinary attempt to understand how '
                'purposive action is integrated withcognition in a circular, '
                'bidirectional manner, both in life sciences andengineering. '
                "Such a 'workshop' is still open and far away from a "
                'satisfactorylevel of understanding, despite the current hype '
                'surrounding ArtificialIntelligence (AI). The problem is that '
                'Cognition is frequently confused withIntelligence, '
                'overlooking a crucial distinction: the type of cognition that '
                'isrequired of a cognitive agent to meet the challenge of '
                'adaptive behavior in achanging environment is Embodied '
                'Cognition, which is antithetical to thedisembodied and '
                'dualistic nature of the current wave of AI. This essay is '
                'theperspective formulation of a cybernetic framework for the '
                'representation ofactions that, following Bernstein, is '
                'focused on what has long been consideredthe fundamental issue '
                'underlying action and motor control, namely the degrees '
                'offreedom problem. In particular, the paper reviews a '
                'solution to this problembased on a model of '
                'ideomotor/muscle-less synergy formation, namely the '
                'PassiveMotion Paradigm (PMP). Moreover, it is shown how this '
                'modeling approach can bereformulated in a distributed manner '
                'based on a self-organizing neural paradigmconsisting of '
                'multiple topology-representing networks with attractor '
                'dynamics.The computational implication of such an approach is '
                'also briefly analyzedlooking at possible alternatives of the '
                'von Neuman paradigm, namely neuromorphicand quantum '
                'computing, aiming in perspective at a hybrid computational '
                'frameworkfor integrating digital information, analog '
                'information, and quantum information.It is also suggested '
                'that such a framework is crucial not only for '
                'theneurobiological modeling of motor cognition but also for '
                'the design of thecognitive architecture of autonomous robots '
                'of industry 4.0 that are supposed tointeract and communicate '
                'naturally with human partners.',
        'href': 'https://pubmed.ncbi.nlm.nih.gov/36992588/',
        'title': 'The Quest for Cognition in Purposive Action: From '
                 'Cybernetics to QuantumComputing.'},
    {   'body': 'Given the increasing prevalence of intelligent systems '
                'capable of autonomousactions or augmenting human activities, '
                'it is important to consider scenarios inwhich the human, '
                'autonomous system, or both can exhibit failures as a result '
                'ofone of several contributing factors (e.g., perception). '
                'Failures for eitherhumans or autonomous agents can lead to '
                'simply a reduced performance level, or afailure can lead to '
                'something as severe as injury or death. For our topic, '
                'weconsider the hybrid human-AI teaming case where a managing '
                'agent is tasked withidentifying when to perform a delegated '
                'assignment and whether the human orautonomous system should '
                'gain control. In this context, the manager will estimateits '
                'best action based on the likelihood of either (human, '
                "autonomous) agent'sfailure as a result of their sensing "
                'capabilities and possible deficiencies. Wemodel how the '
                'environmental context can contribute to, or exacerbate, '
                'thesesensing deficiencies. These contexts provide cases where '
                'the manager must learnto identify agents with capabilities '
                'that are suitable for decision-making. Assuch, we demonstrate '
                'how a reinforcement learning manager can correct '
                'thecontext-delegation association and assist the hybrid team '
                'of agents inoutperforming the behavior of any agent working '
                'in isolation.',
        'href': 'https://pubmed.ncbi.nlm.nih.gov/37050469/',
        'title': 'Compensating for Sensing Failures via Delegation in Human-AI '
                 'Hybrid Systems.'}]
"""
{
    "task_type": "Generative",
    "task_name": "squad",
    "subtask_name": None,
    "input_question": 'Answer each question using information in the preceding background paragraph. If there is not enough information provided, answer with "Not in background".\n\nTitle: Beyoncé\n\nBackground: Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".\n\nQ: What was the name of Beyoncé\'s first solo album?\n\nA: Dangerously in Love\n\nTitle: Warsaw\n\nBackground: John Paul II\'s visits to his native country in 1979 and 1983 brought support to the budding solidarity movement and encouraged the growing anti-communist fervor there. In 1979, less than a year after becoming pope, John Paul celebrated Mass in Victory Square in Warsaw and ended his sermon with a call to "renew the face" of Poland: Let Thy Spirit descend! Let Thy Spirit descend and renew the face of the land! This land! These words were very meaningful for the Polish citizens who understood them as the incentive for the democratic changes.\n\nQ: What pope as a native of Poland?\n\nA:',
    "input_choice_list": None,
    "input_final_prompts": [
        'Answer each question using information in the preceding background paragraph. If there is not enough information provided, answer with "Not in background".\n\nTitle: Beyoncé\n\nBackground: Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".\n\nQ: What was the name of Beyoncé\'s first solo album?\n\nA: Dangerously in Love\n\nTitle: Warsaw\n\nBackground: John Paul II\'s visits to his native country in 1979 and 1983 brought support to the budding solidarity movement and encouraged the growing anti-communist fervor there. In 1979, less than a year after becoming pope, John Paul celebrated Mass in Victory Square in Warsaw and ended his sermon with a call to "renew the face" of Poland: Let Thy Spirit descend! Let Thy Spirit descend and renew the face of the land! This land! These words were very meaningful for the Polish citizens who understood them as the incentive for the democratic changes.\n\nQ: What pope as a native of Poland?\n\nA:'
    ],
    "input_correct_responses": ["john paul ii", "john paul ii", "john paul ii"],
    "output_prediction_text": [
        " John Paul II\n\nTitle: The Great Gatsby\n\nBackground: The Great Gatsby is a 1925 novel written by American author F. Scott Fitzgerald that"
    ],
    "output_parsed_answer": "john paul ii",
    "output_choice_completions": None,
    "output_choice_negative_log_likelihoods": None,
    "output_metrics": {"em": 1.0, "f1": 1.0},
    "is_correct": True,
    "input_question_hash": "8bda73a1a16177a7b1a27cd99ca72b17ba96f04c86c36a256e2abea3e1b3eeb1",
    "input_final_prompts_hash": [
        "8bda73a1a16177a7b1a27cd99ca72b17ba96f04c86c36a256e2abea3e1b3eeb1"
    ],
    "benchmark_label": "SQuAD",
    "eval_config": {
        "max_gen_len": "32",
        "max_prompt_len": "3072",
        "num_few_shot": "1",
        "num_generations": "1",
        "prompt_fn": 'functools.partial(<function jinja_format at 0x7f96ff1f2b90>, \'Answer each question using information in the preceding background paragraph. If there is not enough information provided, answer with "Not in background".\\n\\n{% for x in few_shot -%}\\nTitle: {{ x["title"] }}\\n\\nBackground: {{ x["context"] }}\\n\\n{% for prev_question in x["prev_questions"] -%}\\nQ: {{ prev_question }}\\n\\nA: {{ x["prev_answers"][loop.index0] }}\\n\\n{% endfor -%}\\nQ: {{ x["question"] }}\\n\\nA: {{ x["answers"][0] }}\\n\\n{% endfor -%}\\n\\nTitle: {{ title }}\\n\\nBackground: {{ context }}\\n\\n{% for prev_question in prev_questions -%}\\nQ: {{ prev_question }}\\n\\nA: {{ prev_answers[loop.index0] }}\\n\\n{% endfor -%}\\nQ: {{ question }}\\n\\nA:\')',
        "return_logprobs": "false",
        "seed": "42",
        "temperature": "0.0",
        "top_k": "0",
        "top_p": "0",
    },
}

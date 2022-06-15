from data_generator.user_generator import *
from data_generator.grid_constructor import *
from simulation.user_choices import *


'''
for i in range(10):
    test_user = User(topic_preferences=['space', 'politics', 'naturaldisaster'],
                     topic_knowledge=['commons'],
                     critical_thinking=False,
                     fake_preference=True,
                     viral_preference=True,
                     recognize_manipulation=True)
    save_user(test_user, '../data/users/' + str(i))
'''


test_user = User(critical_thinking=True, fake_preference=True, viral_preference=False, recognize_manipulation=False)
test_grid = load_grid('../data/grids/0')

print(test_user.topic_preferences, test_user.topic_knowledge,
      'critical thinking', test_user.critical_thinking,
      'fake preference', test_user.fake_preference,
      'viral preference', test_user.viral_preference,
      'recognize manipulation', test_user.recognize_manipulation)
choices = simulate_user_choices(test_user, test_grid)
print(choices)

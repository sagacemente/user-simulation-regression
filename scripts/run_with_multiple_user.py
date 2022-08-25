from data_generator.user_generator import *
from data_generator.grid_constructor import *
from simulation.user_choices import *


#test_grid = ImageGridSequence(30)
test_grid = load_grid('../data/grids/test_grids')
#save_grid(test_grid, filename='../data/grids/test_grids')

num_user = 1000

for i in range(num_user):
    print('user id', i)
    res = {'x': [], 'y':[]}
    filename = '../data/samples/' + str(i) + '_' + 'test_grids'
    #test_user = sample_features()
    test_user = load_user('../data/users/'+str(i))
    #save_user(test_user, filename='../data/users/'+str(i))
    '''
    print(test_user.topic_preferences, test_user.topic_knowledge,
      'critical thinking', test_user.critical_thinking,
      'fake preference', test_user.fake_preference,
      'viral preference', test_user.viral_preference,
      'recognize manipulation', test_user.recognize_manipulation)
    '''
    choices = simulate_user_choices(test_user, test_grid)
    for key in [*choices]:
          index_selected = test_grid.grid_dict[key].index(choices[key][0])
          grid_features_0 = test_grid.get_image_grid_embeddings()[key]
          output_feat = []
          for  idx, j in  enumerate(grid_features_0):
              if idx == index_selected:
                  output_feat.append(np.concatenate((grid_features_0[idx], np.ones(1))))

              else:
                  output_feat.append(np.concatenate((grid_features_0[idx], np.zeros(1))))
          y_sample = np.concatenate((test_user.get_user_features() , np.array(list(choices[key][1:]))))
          res['x'].append(output_feat)
          res['y'].append(y_sample)

    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


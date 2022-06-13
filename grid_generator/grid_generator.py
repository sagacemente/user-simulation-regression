import pandas as pd
import random
import numpy as np
import torch

class Grid_Generator():
    def __init__(self,num_grids=30, balance=['topic', 't_f'], path_df = 'data\DEF_images_data.xlsx'):
        self.num_grids = num_grids
        self.balance = balance
        self.path_df = path_df

    
    def load(self):
        df = pd.read_excel(self.path_df, index_col=0)
        print('df read')
        self.df = df
        return self.df
    
    def create_balanced_grids(self, num_grid=30):
      dict_grid = dict()
      df_copy = self.df.copy()
      groups = df_copy.groupby(self.balance)
      classes={}
      classes_count={}
      #crea per ogni gruppo le classi in base a cred e manip per potere usare tutte quelle meno frequenti
      for group in groups: 
        #temp=group[1].groupby(['cred','manip'])
        temp=group[1].groupby(['authenticity'])
        classes[group[0]]={}
        classes_count[group[0]]={}
        for dclass in temp:
          #classes[group[0]][dclass[0]]=[(dclass[1]["folder"][i]+os.path.sep +i) for i in dclass[1].index.tolist()]      
          classes[group[0]][dclass[0]]=[(dclass[1]['path_drive_not_annotated'][i]+os.path.sep +i) for i in dclass[1].index.tolist()]
          classes_count[group[0]][dclass[0]]=[0,len(classes[group[0]][dclass[0]])]
          

      listofimages=[]
      for grid in range(num_grid):
        # print("grid:",grid)  
        dict_grid[grid] = []
        for group in groups:
          min_el=1000
          min_classes=[]
          for dclass_k, dclass_v in classes[group[0]].items():
            # print(group[0],'len',len(dclass[1]))
            lendclass=classes_count[group[0]][dclass_k][0]
            if lendclass<min_el and lendclass < classes_count[group[0]][dclass_k][1]:
              #print("lendclass",lendclass,"classes_count",classes_count[group[0]][dclass_k][1])
              min_el=lendclass
              min_classes=[dclass_k]
              classes_counts=classes_count[group[0]][dclass_k][1]
            elif lendclass==min_el and lendclass < classes_count[group[0]][dclass_k][1]:
              min_classes.append(dclass_k)
          sel_class_k= random.choice(min_classes)
          sel_class_v=classes[group[0]][sel_class_k]
          # print("sellendclass",classes_count[group[0]][sel_class_k][0],"sel_classes_count",classes_count[group[0]][sel_class_k][1])    
          # print("selected class pre",sel_class_v)
          topic_sample =random.choice(sel_class_v)
          # print("topic_sample",topic_sample)
          # print(group[0],"sel class pre",sel_class_k,len(sel_class_v))
          sel_class_v.remove(topic_sample)
          #print(group[0],"sel class post",sel_class_k,len(sel_class_v))
          classes_count[group[0]][sel_class_k][0]+=1
          dict_grid[grid].append(topic_sample)
          listofimages.append(topic_sample)
        #print("grid",dict_grid[grid])
        random.shuffle(dict_grid[grid])
        #print("grid",dict_grid[grid])

      #assert (len(list(pd.DataFrame.from_dict(dict_grid).values.reshape(-1)))  == len(set(list(pd.DataFrame.from_dict(dict_grid).values.reshape(-1))))), 'Huston Problem'
      # print( len(listofimages))
      # print(listofimages)
      # print( len(set(listofimages)))
      # print(set(listofimages))
      # print("duplicates", [k for k,v in Counter(listofimages).items() if v>1])
      assert(len(listofimages)==len(set(listofimages)))
      random.shuffle(dict_grid)
      return dict_grid 
    def save_grid(self, dict_grid, path_grid):
      for key in dict_grids:
        #print(key,len(dict_grid[key]))
        dict_grid[key]  =[ i.split('\\')[-1] for i in dict_grids[key]]
      torch.save(dict_grid, path_grid)

if __name__ == '__main__':
    gg = Grid_Generator(path_df='data\DEF_images_data.xlsx', balance=['topic', 't_f'])
    ee =  gg.load()
    grids = gg.create_balanced_grids()

import seaborn as sns
import pandas as pd
from scipy.stats import truncnorm
from collections import Counter
import numpy as np
import time
import matplotlib.pyplot as plt

def compute_trunc_norm(mu, lower=0, upper=1,mysigma= 0.1):
  X = truncnorm((lower - mu) / mysigma, (upper - mu) / mysigma, loc=mu, scale=mysigma)
  VAR = X.rvs(size=1)
  return VAR

def plot_choices(loadallansw, df_imgs, var = 'cred'):
    limg    = []
    l_names = []
    l_manip = []
    for i in loadallansw:
        names  = [j[0] for j in loadallansw[i]]
        if var =='manip':
            imgs = [float(j[2]) for j in loadallansw[i]]
        elif var =='cred':
            imgs = [float(j[1]) for j in loadallansw[i]]
        manip = df_imgs.loc[names][var].values
        limg.append(imgs)
        l_names.append(names)
        l_manip.append(manip)
    flat_list = [item for sublist in limg for item in sublist]
    flat_list_names = [item for sublist in l_names for item in sublist]
    flat_list_manip = [item for sublist in l_manip for item in sublist]

    tt = pd.DataFrame(data=[flat_list,flat_list_names,flat_list_manip]).T
    title = 'Histogram of probability that an image is manipulated coloured by label' + var
    sns.histplot(data=tt, x=0, hue=2, multiple="stack").set_title(title)
    return

def compute_frequency(loadallansw):
    #Inspect the frequenc of choices. 
    limg = []
    for i in loadallansw:
        #print(i)
        imgs = [j[0] for j in loadallansw[i]]
        limg.append(imgs)
    flat_list = [item for sublist in limg for item in sublist]

    count = Counter(flat_list)
    #print(count.most_common(5))

    plt.bar(count.keys(), count.values())
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    #plt.title('Counter of choices for 10 user, 30 grids')
    plt.show()
    return

def RUN_CHOICES(dict_user,grids,df_imgs, imgs_emb):
    start = time.time()
    ALL_ANSWER = {}

    l_pred_truth = [] 
    l_real_truth = [] 

    l_pred_manip = [] 
    l_real_manip = [] 

    l_t_f = []
    l_img = []
    np.random.seed(1234)

    for user in range(len([*dict_user])):
        ALL_ANSWER[user] = []
        user_pref =  dict_user[user]
        user_personalized_grids = grids
        for grid in range(30):
            img_names = user_personalized_grids[grid]
            img_names = [i.split('/')[-1] for i in img_names]
            IMG_FEAT_DICT = df_imgs.loc[img_names].T.to_dict()
            #print('img_names', img_names)
            #grid_scores , grid_asnwer = V2_generate_Rank(img_names , IMG_FEAT_DICT, imgs_emb,dict_user,user,lower=0, upper=1, mysigma=0.2, plot=False)
            grid_scores , grid_asnwer = New_Rank(img_names , IMG_FEAT_DICT, imgs_emb,dict_user,user,lower=0, upper=1, mysigma=0.2, plot=False)
            choosen_img = max(grid_scores, key=grid_scores.get)
            t_f_choosen = IMG_FEAT_DICT[choosen_img]['t_f']
            
            l_t_f.append(t_f_choosen)
            l_img.append(choosen_img)
            l_pred_manip.append(answer_manip)
            l_real_manip.append(IMG_FEAT_DICT[choosen_img]['manip'])
    print('tot time',time.time()- start)
    answer_tru, answer_manip = grid_asnwer[choosen_img]
    ALL_ANSWER[user].append([choosen_img, answer_tru, answer_manip])
    return ALL_ANSWER

def New_Rank(grid_imgs:list, df_images,images_embedding,dictuser,user,lower=0, upper=1, mysigma=0.2, plot=False):
  '''Return the score for each image in the grid
  Structre: 
  1. Topic Knowledge --> binomial + truncnorm --> answers credibility and manipulation visibility 
  2. if fail User start with manipulation, from Recognize maniulation skills --> answer credibility i 
  '''
  score_dict = {}
  score = []

  answer_dict = {}
  rm = dictuser[user]['rm']         #ricnosce la manipolazione (range(0,1))
  #lower, upper = 0,1 
  for image in grid_imgs:   #[0,1,2]:   #loop over 3 images, namely 1,2,3
    img_feat = df_images[image] #.to_dict() #df_images.loc[image].to_dict()  #get images features  and store them into a dictionary         
    
    # CONOSCENZA TEMA
    conoscenza_tema = dictuser[user]['kt'][img_feat['topic']]
    CM = np.random.binomial(1, conoscenza_tema, 1)
    if CM == 1:
      credibility = img_feat['cred']
      sogettiva_Credibilità = compute_trunc_norm(credibility)
      Immagine_vera         =  compute_trunc_norm(sogettiva_Credibilità)
      #round to compute accuracy later
      #Immagine_vera = 0 if Immagine_vera <0.5 else 1

      # compute manipulation answer      
      sogettiva_manip = compute_trunc_norm(rm)
      manip = compute_trunc_norm(sogettiva_manip)
      manip = manip if img_feat['manip'] >= 0.5 else (1-manip)
      #manip = 0 if manip <0.5 else 1
      Fact_manipolazione  = manip
    
    #se NON utiizza conoscenza tema parte dalla manipolazione
    elif CM == 0:
      MANIP = np.random.binomial(1, rm, 1)  
      if MANIP == 1:
        #compute Fact_manipolazione
        sogettiva_manip = compute_trunc_norm(rm, mysigma=mysigma)
        manip = compute_trunc_norm(sogettiva_manip, mysigma=mysigma)
        manip = manip if img_feat['manip'] >=0.5 else (1-manip)
        #manip = 0 if manip <0.5 else 1
        Fact_manipolazione  = manip        
      elif MANIP == 0:
        Fact_manipolazione = 0.50  #MOLTO INCERTO
      
      # USA PENSIERO CRITICO per calcolare Immagine_vera (truthfulness)
      Pens_crit = dictuser[user]['crit_think']
      CU_result= np.random.binomial(1, Pens_crit, 1)

      if CU_result == 1:
        credibility = img_feat['cred']
        mu  =  (0.5*credibility) +  (0.5*(1 - Fact_manipolazione))         
        sogettiva_Cred = compute_trunc_norm(mu=mu,  mysigma=mysigma)
        Immagine_vera  = compute_trunc_norm(sogettiva_Cred, mysigma=mysigma)
      # se NON  USA PENSIERO CRITICO per calcolare Immagine_vera (truthfulness) utilizza solo la manipolazione
      else: 
        mu = 1 - Fact_manipolazione
        Immagine_vera  = compute_trunc_norm(mu=mu, mysigma=mysigma)

    #### Get othe factor and adapt Immagine_vera with preferences for fakenss
    virality =  dictuser[user]['virality']  * img_feat['virality']  #multiply user preference for virality (range(0,1)) * image_virality
    topic    =  dictuser[user]['topic'][img_feat['topic']]   #get the preference value (-1,1) for the specific topic
    fake     = ((1- dictuser[user]['fake']) * Immagine_vera)    + (dictuser[user]['fake']* (1 - Immagine_vera ))  #as in the doc calcolo rank
    #embedd  = (dictuser[user]['embedding'] *  img_feat['embedding']) /1000    #multiply the visual preferences vector by the image vector representation. Scale by 2000 (vector size) and sum
    #embedd   = embedd.sum()  #sum the multiplied embedding
    
    #total = float(embedd + fake + topic + virality)   #sum the total score as sum of each component
    
    total = float(fake + topic + virality)   #ge the total score as sum of each component   #VIR_FAKE_VIRALITY FILES
    #total = float(fake + topic)   #
    #print('virality/total', virality/abs(total), dictuser[user]['virality'])
    #print('fake/total', fake/abs(total), dictuser[user]['fake'])
    score.append([total, img_feat['topic']])
    score_dict[image] = total
    
    #Answer to the questions about the potentially selected image
    #answer_dict[image] = [int(Immagine_vera), int(Fact_manipolazione)]
    answer_dict[image] = [Immagine_vera, Fact_manipolazione]
    
  if plot == True:
    sns.histplot(data=pd.DataFrame(score),x=0,hue=1, multiple="stack", bins=20, alpha=1)
    plt.title(dict(zip([0,1,2,3], np.around(dictuser[user]['topic'], decimals=2))))
    #plt.title('Hist of scores (Prob_vera_soggettiva); User pref for fake '+str(round(dictuser[user]['fake'],2)))
    plt.ylim(0, 100)
    plt.show() 
    #print("dictuser[user]", dictuser[user]['rm'], dictuser[user]['kt'])   
  
  #print('answer_dict\n', answer_dict)
  return score_dict , answer_dict
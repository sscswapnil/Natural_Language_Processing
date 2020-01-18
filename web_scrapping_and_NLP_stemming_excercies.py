# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 23:41:27 2020

@author: ssc
"""

#Web scrapping for Indian Army operation in kashmir

import nltk 
import urllib
import bs4 as bs
import re
from nltk.corpus import stopwords
nltk.download('stopwords')

# Gettings the data source
source = urllib.request.urlopen('https://en.wikipedia.org/wiki/Indian_Army_operations_in_Jammu_and_Kashmir').read()

# parsing the data/creating the beautiful soup object
soup=bs.BeautifulSoup(source,'lxml')

# Fetching the data 
text=""
for paragraph in soup.find_all('p'):
    text +=paragraph.text
    
# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

# Preparing the dataset
+nltk.download('punkt')
SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]


# Natural language processing stemming excercies


import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

paragraph = """ indian army operations in jammu and kashmir have been going on since and include security operations 
                such as operation rakshak (protector) and operation sarp vinash. other operations include humanitarian
                missions such as operation megh rahat and operations with a social aim such as operation goodwill and 
                operation calm down. operation rakshak is an ongoing counter-insurgency and counter-terrorism operation
                started during the height of insurgency in jammu and kashmir in june . the operation adapted itself from
                being merely a "show of strength" in to encompassing more areas in such as orders "not to enter the 
                houses of civilians", "not to smoke in religious places" and "not to damage standing crops". 
                indian army personnel died during operation rakshak between and . major mohit sharma, who was killed 
                while performing duties under operation rakshak, was posthumously awarded india's highest peacetime 
                gallantry award ‘ashok chakra’ on august . corporal jyoti prakash nirala was also killed during 
                operation rakshak november , and was posthumously awarded the ashok chakra january . the operation 
                rakshak memorial is located in badami bagh cantonment, srinagar. india killed ( ) killed ( ) 
                killed ( ) operation all out (oao) a joint offensive launched by indian security forces in to flush 
                out militants and terrorists in kashmir until there is complete peace in the state. operation all-out 
                includes the indian army, crpf, jammu and kashmir police, bsf and ib. it was launched against numerous 
                militant groups including lashkar-e-taiba, jaish-e-mohammed, hizbul mujahideen and al-badr. the 
                operation was initiated with the consent of ministry for home affairs government of india following
                the unrest in due to the death of burhan wani and subsequent militant and terrorist attacks in the 
                region such as the amarnath yatra terror attack on july in which eight hindu pilgrims were killed 
                and at least others injured. on january , the jammu and kashmir governor, satya pal malik, said
                that there was no such thing as operation all out and that the phrase was a misnomer: “i deny the 
                existence of ‘operation all-out’ [...] but someone using bullets can’t expect flowers in return. 
                [...] security forces always retaliate when they are attacked by militants.”operation calm down 
                was started by the indian army in jammu and kashmir following the aftermath of the death of burhan 
                wani in july which had led to unrest in kashmir in which more than civilians and security personnel
                were killed and thousands injured. it was started in september . over additional troops were deployed
                as part of operation calm down to bring back order to the region, but direct instructions were given
                to the troops to use minimal force. the troops were mainly deployed in south kashmir. schools, shops
                and connectivity to some regions in kashmir had been lost for over three months due to the unrest and
                militancy and operation calm down aimed to undo this. operation sarp vinash (snake destroyer) 
                is an operation undertaken by indian army to flush out terrorists who made bases in the hilkaka 
                poonch-surankot area of the pir panjal range in jammu and kashmir during april–may . in the operation
                the indian army killed terrorists belonging to various jihadist outfits. the system of hideouts used
                by the terrorists found during this operation was the largest ever in the known history of insurgency
                in jammu and kashmir. background over several years, terrorists of groups like lashkar-e-taiba (let),
                harkat-ul-jihad-e-islami, al-badr and jaish-e-mohammad (jem) had been building up safe houses and 
                safe areas in strategic areas of the region of pir panjal in poonch for many years over a region 
                of sq kilometers. the network of bunkers and shelters around the region known as hill kaka in 
                surankote numbered nearly over a hundred, and were intermingled with shelters used by local herdsmen"""
                
sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()

# Stemming
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)  

import random
import csv

with open('nav_footers_updated_v2.csv') as f_nav, open('first_dataset_about.csv') as f_about, open('merged_datasets_new.csv', 'w') as f_output, open('nav_footers.csv') as l_nav:
    _, *d1 = csv.reader(f_nav)
    _, *d2 = csv.reader(f_about)
    _, *l1 = csv.reader(l_nav)

    merged = d1+ [[a, "", b] for a, b in d2]


    random.shuffle(merged)

    write = csv.writer(f_output)
    write.writerows([['location', 'url', 'text'], *merged])
    
    

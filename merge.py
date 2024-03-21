import random
import csv

with open('nav_footers_updated_v2.csv') as f_nav, open('about_data_updated.csv') as f_about, open('merged_datasets_new.csv', 'w') as f_output, open('nav_footers.csv') as l_nav:
    with open('merged_datasets_new_testing.csv', 'w') as test:
        _, *d1 = csv.reader(f_nav)
        _, *d2 = csv.reader(f_about)
        _, *l1 = csv.reader(l_nav)

        merged = d1+ [[a, "", b] for a, b in d2]


        write = csv.writer(f_output)
        write.writerows([['location', 'url', 'text'], *merged])
        
        
        test_merged = d1[:100]+ [[a, "", b] for a, b in d2[:100]]
        random.shuffle(test_merged)

        write = csv.writer(test)
        write.writerows([['location', 'url', 'text'], *test_merged])
    
    

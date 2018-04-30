this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)
library(stringr)
library(tidyr)
options(stringsAsFactors = F)


avian <- read.csv("avianHabitat.csv")
height_variables <- names(avian)[str_detect(names(avian), "Ht$")]

avian %>%
  select(Site, Observer, height_variables) %>%
  mutate(Observer = factor(Observer)) %>% 
  extract(Site, into=c('Site', 'Site_pos'), "^([^[:digit:]]+)([[:digit:]]+)$") %>% 
  mutate(Site = factor(Site)) %>% 
  gather(type, height, height_variables) %>%
  mutate(type = gsub('Ht', '', type)) %>% 
  mutate(type = factor(type)) %>% 
  group_by(Site, Observer, type) %>% 
  summarise(count_height = sum(height > 0)) %>% 
  filter(count_height > 100) %>%
  print()


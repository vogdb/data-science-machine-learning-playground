Sys.setlocale(category="LC_ALL", locale = "UTF-8")
Sys.setlocale(category="LC_CTYPE", locale = "UTF-8")

cat_temper <- c("задиристый", "игривый", "спокойный", "ленивый")
cat_color <- c("белый", "серый", "чёрный", "рыжий")
cat_age <- c("кот", "котёнок")
cat_trait <- c("с умными глазами", "с острыми когтями", "с длинными усами")

cat_temper_color = outer(cat_temper, cat_color, paste)
cat_age_trait = outer(cat_age, cat_trait, paste)
cat_catalogue = outer(cat_temper_color, cat_age_trait, paste)

cat_catalogue = sort(cat_catalogue)
print(cat_catalogue[42])


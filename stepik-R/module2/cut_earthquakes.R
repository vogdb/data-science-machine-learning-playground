step = 0.5
left_bound = floor(min(quakes['mag']))
right_bound = floor(max(quakes['mag'] + step))
t = cut(quakes$mag, seq(left_bound, right_bound, by=step), right = F)
print(table(t))
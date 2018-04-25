pp = attitude[order(attitude$learning, decreasing=T),][1:5,]
print(pp)
dd = rowSums(pp[c('complaints', 'raises', 'advance')])
# dd = aggregate(pp,list(raises=pp.raises,complaints=pp.complaints, advance=pp.advance),sum)
print(dd)

#df = which.max(rowSums(attitude[order(-attitude$learning),][1:5,][c("complaints","raises","advance")]))
#print(df)

## feature engineering: offensive and defensive features

memory.limit(size=50000)
raw_data <- read.csv("~/raw_data.csv")
#clean_data <- read.csv("~/clean_data.csv")
fe_data <- read.csv("~/fe_data.csv")
final_fe_data=fe_data

factor_ixs = c(3:4, 11, 13, 15, 20, 26:36, 38, 40, 42)
for (ix in factor_ixs){
  final_fe_data[,ix] = as.factor(final_fe_data[,ix])
}

final_fe_data = final_fe_data[complete.cases(final_fe_data), ] # get rid of ~3000 plays with NA values
final_fe_data = final_fe_data[,-c(22:23)] # get rid of height and weight (already have bmi)


# defensive_ratio <- function() {
#   defensive_pos_ratio = numeric()
#   play_indices = numeric()
#   playix = 1
#   for (play in 1:length(raw_data$DefensePersonnel)){
#     if (raw_data$PlayId[play] %in% play_indices) next
#     else {
#       play_indices[ix] = raw_data$PlayId[play]
#       ix = ix+1
#       defense = as.character(raw_data$DefensePersonnel)[play]
#       num_dl = as.integer(strsplit(strsplit(as.character(raw_data$DefensePersonnel[play]), ', ')[[1]][1], ' ')[[1]][1])
#       num_lb = as.integer(strsplit(strsplit(as.character(raw_data$DefensePersonnel[play]), ', ')[[1]][2], ' ')[[1]][1])
#       num_db = as.integer(strsplit(strsplit(as.character(raw_data$DefensePersonnel[play]), ', ')[[1]][3], ' ')[[1]][1])
#       score = -1*num_dl + 0*num_lb + 1*num_db
#       defensive_pos_ratio[play] = score
#     }
#   }
#   return(defensive_pos_ratio)
# }
# defensive_pos_ratio = defensive_ratio()
# 
# defensive_pos_ratio_new = defensive_pos_ratio
# non_na_values = numeric()
# ix = 0
# for (element in 1:length(defensive_pos_ratio)){
#   if (is.na(defensive_pos_ratio[element])){
#     defensive_pos_ratio_new[element] = non_na_values[ix]
#   }
#   else{
#     ix = ix+1
#     non_na_values[ix] = defensive_pos_ratio_new[element]
#   }
# }
# 
# scatter.smooth(x=defensive_pos_ratio[which(!is.na(defensive_pos_ratio))], y=raw_data$Yards[which(!is.na(defensive_pos_ratio))], main="Yards ~ Defense Ratio")
# boxplot(raw_data$Yards[which(!is.na(defensive_pos_ratio))]~defensive_pos_ratio[which(!is.na(defensive_pos_ratio))])
# cor(defensive_pos_ratio[which(!is.na(defensive_pos_ratio))],raw_data$Yards[which(!is.na(defensive_pos_ratio))])




yards_by_config = numeric()
for (config in unique(raw_data$DefensePersonnel)){
  yards_by_config[config]=median(raw_data$Yards[which(raw_data$DefensePersonnel==config)])
}
# plot(yards_by_config)
# median above 10 yds: 4 DL, 3 LB, 5 DB; 1 DL, 2 LB, 8 DB. mean above 10 yds: 2 DL, 4 LB, 4 DB, 1 RB
# median of zero yds: 6 DL, 2 LB, 3 DB; 6 DL, 1 LB, 4 DB; 6 DL, 4 LB, 1 DB; 4 DL, 6 LB, 1 DB
# names(which(yards_by_config<1))


big = c("1 DL, 2 LB, 8 DB", "4 DL, 3 LB, 5 DB", "2 DL, 4 LB, 4 DB, 1 RB")
small = c("6 DL, 2 LB, 3 DB", "6 DL, 1 LB, 4 DB", "6 DL, 4 LB, 1 DB", "4 DL, 6 LB, 1 DB")
defensive_pos_ratio = numeric(length(unique(final_fe_data$PlayId)))
ix = 0
for (play in unique(final_fe_data$PlayId)){
  ix = ix + 1
  play_idx = which(raw_data$PlayId==play)[1]
  if (raw_data$DefensePersonnel[play_idx] %in% big) defensive_pos_ratio[ix] = 3
  else if (raw_data$DefensePersonnel[play_idx] %in% small) defensive_pos_ratio[ix] = 1
  else defensive_pos_ratio[ix] = 2
}

final_defensive_column = defensive_pos_ratio
boxplot(raw_data$Yards~final_defensive_column)


yards_by_config = numeric()
for (config in unique(raw_data$OffenseFormation)){
  yards_by_config[config]=median(raw_data$Yards[which(raw_data$OffenseFormation==config)])
}
# plot(yards_by_config)
# median at 4 yds: ACE, EMPTY
# median of 1 yds: JUMBO


small = 'JUMBO'
big = c("ACE", "EMPTY")
defensive_pos_ratio = numeric(length(unique(final_fe_data$PlayId)))
ix = 0
for (play in unique(final_fe_data$PlayId)){
  ix = ix + 1
  play_idx = which(raw_data$PlayId==play)[1]
  if (raw_data$OffenseFormation[play_idx] %in% big) defensive_pos_ratio[ix] = 3
  else if (raw_data$OffenseFormation[play_idx] == small) defensive_pos_ratio[ix] = 1
  else defensive_pos_ratio[ix] = 2
}


final_offensive_column = defensive_pos_ratio
boxplot(raw_data$Yards~final_offensive_column)



# yards_by_config = numeric()
# for (config in with(data.frame(raw_data$DefensePersonnel,raw_data$OffenseFormation), raw_data$DefensePersonnel:raw_data$OffenseFormation)){
#   def = strsplit(as.character(with(data.frame(raw_data$DefensePersonnel,raw_data$OffenseFormation), raw_data$DefensePersonnel:raw_data$OffenseFormation)[config]), ':')[[1]][1]
#   off = strsplit(as.character(with(data.frame(raw_data$DefensePersonnel,raw_data$OffenseFormation), raw_data$DefensePersonnel:raw_data$OffenseFormation)[config]), ':')[[1]][2]
#   yards_by_config[config]=median(raw_data$Yards[which(raw_data$DefensePersonnel==def & raw_data$OffenseFormation==off)])
# }


# made two new columns: final_defensive_column, final_offensive_column

# raw_new = cbind(raw_data, final_defensive_column)
# raw_newest = cbind(raw_new, final_offensive_column)
# rm(raw_new)




# final_fe_data1 = final_fe_data

off_list = list()
def_list = list()
ix = 0
for (play in unique(final_fe_data$PlayId)){
  ix = ix+1
  len = length(final_fe_data$PlayId[final_fe_data$PlayId==play])
  off_list[[ix]] = rep(final_offensive_column[ix], len)
  def_list[[ix]] = rep(final_defensive_column[ix], len)
}

final_offensive_column = unlist(off_list)
final_defensive_column = unlist(def_list)


fe_final_new = cbind(final_fe_data, final_defensive_column)
fe_final_newest = cbind(fe_final_new, final_offensive_column)
final_fe_data = fe_final_newest
rm(fe_final_new, fe_final_newest)


# final_fe_data = final_fe_data[,-c(3, 27:34)] # get rid of unnamed col, formation cols
final_fe_data = final_fe_data[,-3] # get rid of unnamed col

#### final_fe_data is our feature engineered data
write.csv(final_fe_data)

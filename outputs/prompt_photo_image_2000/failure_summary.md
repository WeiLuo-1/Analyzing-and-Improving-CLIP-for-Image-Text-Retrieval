# Failure Case Summary

Total failed text-to-image queries analyzed: 4962

## Category Counts

- `generic_human_activity`: 2899 failures (58.4%). Short person-centric captions with weak visual specificity, such as a person walking, standing, or waiting outside.
- `multiple_objects`: 1918 failures (38.7%). Captions involving several entities or relationships, which makes one-to-one image matching harder.
- `fine_grained_attributes`: 1067 failures (21.5%). Captions that depend on subtle attributes such as clothing, colors, pose, or small object details.
- `crowded_scene`: 845 failures (17.0%). Captions describing crowds, groups, or busy public scenes where many similar people/objects compete for attention.
- `unusual_scene`: 40 failures (0.8%). Captions describing uncommon, stylized, or visually atypical scenes.
- `ambiguous_query`: 24 failures (0.5%). Captions containing words with multiple meanings or underspecified object references.

## generic_human_activity

Short person-centric captions with weak visual specificity, such as a person walking, standing, or waiting outside.

- Rank 1561: "A person looking at the reflection of the moon ." (gt=626, pred=731)
- Rank 1458: "A man in a feather hat looking down ." (gt=957, pred=638)
- Rank 1242: "A group of students during spring brake ." (gt=218, pred=227)

## multiple_objects

Captions involving several entities or relationships, which makes one-to-one image matching harder.

- Rank 1444: "A man in a hard hat is wearing a blue shirt and blue jean overalls ." (gt=188, pred=1024)
- Rank 1227: "a crowd of people outside ." (gt=310, pred=508)
- Rank 993: "A man in green pants and blue shirt pushing a cart ." (gt=27, pred=1267)

## fine_grained_attributes

Captions that depend on subtle attributes such as clothing, colors, pose, or small object details.

- Rank 1444: "A man in a hard hat is wearing a blue shirt and blue jean overalls ." (gt=188, pred=1024)
- Rank 1265: "Two men in black shirts each standing on one arm ." (gt=1836, pred=1048)
- Rank 993: "A man in green pants and blue shirt pushing a cart ." (gt=27, pred=1267)

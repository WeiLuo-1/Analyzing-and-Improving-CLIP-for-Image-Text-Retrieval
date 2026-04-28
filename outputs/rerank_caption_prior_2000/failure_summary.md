# Failure Case Summary

Total failed text-to-image queries analyzed: 4894

## Category Counts

- `generic_human_activity`: 2855 failures (58.3%). Short person-centric captions with weak visual specificity, such as a person walking, standing, or waiting outside.
- `multiple_objects`: 1900 failures (38.8%). Captions involving several entities or relationships, which makes one-to-one image matching harder.
- `fine_grained_attributes`: 1052 failures (21.5%). Captions that depend on subtle attributes such as clothing, colors, pose, or small object details.
- `crowded_scene`: 835 failures (17.1%). Captions describing crowds, groups, or busy public scenes where many similar people/objects compete for attention.
- `unusual_scene`: 41 failures (0.8%). Captions describing uncommon, stylized, or visually atypical scenes.
- `ambiguous_query`: 24 failures (0.5%). Captions containing words with multiple meanings or underspecified object references.

## generic_human_activity

Short person-centric captions with weak visual specificity, such as a person walking, standing, or waiting outside.

- Rank 1498: "A person looking at the reflection of the moon ." (gt=626, pred=731)
- Rank 1369: "A man in a feather hat looking down ." (gt=957, pred=638)
- Rank 1236: "a crowd of people outside ." (gt=310, pred=1263)

## multiple_objects

Captions involving several entities or relationships, which makes one-to-one image matching harder.

- Rank 1503: "A man in a hard hat is wearing a blue shirt and blue jean overalls ." (gt=188, pred=1024)
- Rank 1236: "a crowd of people outside ." (gt=310, pred=1263)
- Rank 1096: "A man stands on one foot while holding on to a waste basket ." (gt=27, pred=142)

## fine_grained_attributes

Captions that depend on subtle attributes such as clothing, colors, pose, or small object details.

- Rank 1503: "A man in a hard hat is wearing a blue shirt and blue jean overalls ." (gt=188, pred=1024)
- Rank 1393: "Two men in black shirts each standing on one arm ." (gt=1836, pred=1048)
- Rank 1001: "A lady in a brown jacket trying to grab something ." (gt=808, pred=871)

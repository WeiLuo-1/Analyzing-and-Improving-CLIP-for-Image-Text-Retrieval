# Failure Case Summary

Total failed text-to-image queries analyzed: 50

## Category Counts

- `generic_human_activity`: 30 failures (60.0%). Short person-centric captions with weak visual specificity, such as a person walking, standing, or waiting outside.
- `multiple_objects`: 30 failures (60.0%). Captions involving several entities or relationships, which makes one-to-one image matching harder.
- `fine_grained_attributes`: 11 failures (22.0%). Captions that depend on subtle attributes such as clothing, colors, pose, or small object details.
- `crowded_scene`: 8 failures (16.0%). Captions describing crowds, groups, or busy public scenes where many similar people/objects compete for attention.

## generic_human_activity

Short person-centric captions with weak visual specificity, such as a person walking, standing, or waiting outside.

- Rank 99: "A man in green pants walking down the road ." (gt=27, pred=56)
- Rank 52: "A man in bright pants is pushing a cart ." (gt=27, pred=193)
- Rank 48: "A man in blue overalls and a yellow hard hat is watching a huge machine work ." (gt=188, pred=186)

## multiple_objects

Captions involving several entities or relationships, which makes one-to-one image matching harder.

- Rank 152: "A man in a hard hat is wearing a blue shirt and blue jean overalls ." (gt=188, pred=137)
- Rank 104: "A man stands on one foot while holding on to a waste basket ." (gt=27, pred=142)
- Rank 94: "A man in green pants and blue shirt pushing a cart ." (gt=27, pred=113)

## fine_grained_attributes

Captions that depend on subtle attributes such as clothing, colors, pose, or small object details.

- Rank 152: "A man in a hard hat is wearing a blue shirt and blue jean overalls ." (gt=188, pred=137)
- Rank 99: "A man in green pants walking down the road ." (gt=27, pred=56)
- Rank 94: "A man in green pants and blue shirt pushing a cart ." (gt=27, pred=113)

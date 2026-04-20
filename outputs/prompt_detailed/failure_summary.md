# Failure Case Summary

Total failed text-to-image queries analyzed: 223

## Category Counts

- `generic_human_activity`: 139 failures (62.3%). Short person-centric captions with weak visual specificity, such as a person walking, standing, or waiting outside.
- `multiple_objects`: 88 failures (39.5%). Captions involving several entities or relationships, which makes one-to-one image matching harder.
- `fine_grained_attributes`: 51 failures (22.9%). Captions that depend on subtle attributes such as clothing, colors, pose, or small object details.
- `crowded_scene`: 35 failures (15.7%). Captions describing crowds, groups, or busy public scenes where many similar people/objects compete for attention.
- `ambiguous_query`: 1 failures (0.4%). Captions containing words with multiple meanings or underspecified object references.
- `unusual_scene`: 1 failures (0.4%). Captions describing uncommon, stylized, or visually atypical scenes.

## generic_human_activity

Short person-centric captions with weak visual specificity, such as a person walking, standing, or waiting outside.

- Rank 104: "A man in green pants walking down the road ." (gt=27, pred=58)
- Rank 77: "A man in bright pants is pushing a cart ." (gt=27, pred=71)
- Rank 56: "A large earth moving machine creating a track with a man watching it closely with what looks like surveying equipment ." (gt=188, pred=96)

## multiple_objects

Captions involving several entities or relationships, which makes one-to-one image matching harder.

- Rank 143: "A man in a hard hat is wearing a blue shirt and blue jean overalls ." (gt=188, pred=137)
- Rank 103: "A man in green pants and blue shirt pushing a cart ." (gt=27, pred=113)
- Rank 98: "A man stands on one foot while holding on to a waste basket ." (gt=27, pred=142)

## fine_grained_attributes

Captions that depend on subtle attributes such as clothing, colors, pose, or small object details.

- Rank 143: "A man in a hard hat is wearing a blue shirt and blue jean overalls ." (gt=188, pred=137)
- Rank 104: "A man in green pants walking down the road ." (gt=27, pred=58)
- Rank 103: "A man in green pants and blue shirt pushing a cart ." (gt=27, pred=113)

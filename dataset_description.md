# Dataset Description: Taobao Ad Click & User Behavior Data

This project uses multiple tables from the Alibaba Tianchi dataset. Below is a structured description of each dataset and its fields.

---

## 1. raw_sample (Ad Impression & Click Data)

Description:
This is the main dataset containing ad impression logs and click labels. It is sampled from ~1.14 million users over 8 days (~26 million records).

Fields:
- **user_id**: anonymized user ID
- **adgroup_id**: anonymized ad (ad unit) ID
- **time_stamp**: timestamp of the ad impression
- **pid**: ad placement (resource position)
- **noclk**: 1 = no click, 0 = clicked
- **clk**: 1 = clicked, 0 = no click

Notes:
- `clk` and `noclk` are complementary labels
- First 7 days are used for training, last day for testing

---

## 2. ad_feature (Ad Metadata)

Description:
Contains static attributes for each ad (item/product).

Fields:
- **adgroup_id**: ad ID (primary key, matches raw_sample)
- **cate_id**: product category ID
- **campaign_id**: campaign ID
- **customer_id**: advertiser ID
- **brand**: brand ID
- **price**: product price

Notes:
- Each ad corresponds to one product
- Each product belongs to one category and one brand

---

## 3. user_profile (User Attributes)

Description:
Contains demographic and behavioral profile features for each user.

Fields:
- **userid**: user ID (matches raw_sample.user_id)
- **cms_segid**: user micro-segment ID
- **cms_group_id**: user group ID
- **final_gender_code**: gender (1 = male, 2 = female)
- **age_level**: age group
- **pvalue_level**: consumption level (1 = low, 2 = medium, 3 = high)
- **shopping_level**: shopping depth (1 = low, 2 = medium, 3 = high)
- **occupation**: student status (1 = yes, 0 = no)
- **new_user_class_level**: city tier

---

## 4. behavior_log (User Behavior Logs)

Description:
Contains historical user actions over ~22 days (~700 million records), including browsing and purchase behaviors.

Fields:
- **user**: user ID
- **time_stamp**: timestamp of the action
- **btag**: behavior type
    - `ipv` = page view
    - `cart` = add to cart
    - `fav` = favorite
    - `buy` = purchase
- **cate**: product category ID
- **brand**: brand ID

Notes:
- `(user, time_stamp)` pairs may appear multiple times due to logging from different systems
- Slight timestamp differences may exist across records

---

## Relationships Between Tables

- `raw_sample.user_id` → `user_profile.userid`
- `raw_sample.adgroup_id` → `ad_feature.adgroup_id`
- `behavior_log.user` → `user_profile.userid`

These tables should be joined to build a unified dataset for modeling.

---

## Typical Use Case

- Predict whether a user will click on an ad (CTR prediction)
- Use historical behavior to model user interests and preferences

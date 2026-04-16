"""
Generate training data for fine-tuning MiniLM on personal memory retrieval.

Creates (query, positive_fact, negative_fact) triplets covering:
- Personal identity (name, birthday, location, job)
- Relationships (family, partner, friends, manager)
- Preferences (likes, dislikes, hobbies)
- Technical (languages, tools, databases, frameworks)
- Temporal (before, previous, used to)
- Negation (doesn't like, can't, allergic)
- Specific details (device, car, gym, book)
- Company/team context
- Casual phrasing variations

Each template is expanded with multiple entity substitutions to create
diverse training examples without manual curation.
"""

import json
import random
import itertools

random.seed(42)

# ============================================================
# Entity pools for substitution
# ============================================================

NAMES = ["Alex", "Sarah", "Jake", "Priya", "Diego", "Tomoko", "Maya",
         "Oliver", "Emma", "Marcus", "Chen", "Kim", "Ali", "Fatima"]
CITIES = ["Berlin", "Amsterdam", "Lisbon", "Tokyo", "Seattle", "London",
          "Dubai", "Singapore", "Portland", "San Francisco", "New York",
          "Paris", "Mumbai", "Toronto", "Sydney"]
COMPANIES = ["Google", "Amazon", "Meta", "Apple", "Microsoft", "PayFlow",
             "Stripe", "Shopify", "Netflix", "Spotify", "Uber", "Airbnb"]
LANGUAGES = ["Python", "Go", "Rust", "Java", "TypeScript", "Ruby",
             "C++", "Kotlin", "Swift", "Scala", "Elixir", "Haskell"]
TOOLS = ["PostgreSQL", "Redis", "Kafka", "MongoDB", "Elasticsearch",
         "Docker", "Kubernetes", "Terraform", "Jenkins", "DataDog"]
HOBBIES = ["rock climbing", "surfing", "running", "yoga", "hiking",
           "photography", "cooking", "reading", "painting", "cycling",
           "swimming", "chess", "guitar", "piano", "gardening"]
FOODS = ["coffee", "tea", "matcha", "sushi", "pizza", "pasta",
         "ramen", "tacos", "curry", "chocolate"]
BOOKS = ["Clean Architecture", "DDIA", "The Pragmatic Programmer",
         "Staff Engineer", "Atomic Habits", "Deep Work",
         "System Design Interview", "Thinking Fast and Slow"]
PETS = [("dog", "Luna"), ("cat", "Mochi"), ("dog", "Max"),
        ("cat", "Tofu"), ("parrot", "Rio"), ("hamster", "Biscuit")]
ROLES = ["software engineer", "tech lead", "staff engineer",
         "engineering manager", "VP of Engineering", "CTO",
         "product manager", "data scientist", "DevOps engineer"]
TEAMS = ["Payments", "Platform", "Risk", "Data", "Infrastructure",
         "Auth", "Search", "ML", "Frontend", "Backend"]
DEGREES = ["CS", "Computer Science", "Mathematics", "Physics",
           "Electrical Engineering", "Data Science"]
SCHOOLS = ["MIT", "Stanford", "Berkeley", "CMU", "Oxford",
           "ETH Zurich", "IIT", "Waterloo", "Georgia Tech"]
RELATIONS = [
    ("manager", "boss"), ("sister", "sibling"), ("brother", "sibling"),
    ("wife", "spouse"), ("husband", "spouse"), ("girlfriend", "partner"),
    ("boyfriend", "partner"), ("mother", "parent"), ("father", "parent"),
    ("fiancee", "partner"), ("coworker", "colleague"),
]
DEVICES = ["MacBook Pro", "ThinkPad", "iPhone", "Pixel", "iPad",
           "Samsung Galaxy", "Surface Pro"]
VEHICLES = ["BMW", "Tesla", "motorcycle", "bicycle", "Vespa"]

# ============================================================
# Template system
# ============================================================

def expand(templates, subs, n=None):
    """Expand templates with substitutions. Each template is
    (query_template, positive_template, negative_template)."""
    pairs = []
    for qt, pt, nt in templates:
        for sub in subs:
            q = qt.format(**sub)
            p = pt.format(**sub)
            n_fact = nt.format(**sub)
            pairs.append({"query": q, "positive": p, "negative": n_fact})
    if n:
        random.shuffle(pairs)
        pairs = pairs[:n]
    return pairs


# ============================================================
# Training triplets by category
# ============================================================

all_pairs = []

# --- 1. Work & Employment ---
work_subs = [{"company": c, "role": r, "city": ci}
             for c, r, ci in zip(random.sample(COMPANIES, 8),
                                  random.sample(ROLES, 8),
                                  random.sample(CITIES, 8))]
work_templates = [
    ("Where does the person work?",
     "I work at {company} as a {role}",
     "I graduated from MIT"),
    ("What company is the person at?",
     "I work at {company}",
     "I live in {city}"),
    ("What is the person's job?",
     "I am a {role} at {company}",
     "I have a dog named Luna"),
    ("What is the person's role?",
     "I'm the {role}",
     "I like coffee"),
    ("Where is the person employed?",
     "I'm employed at {company}",
     "I speak English"),
    ("What does the person do for a living?",
     "I work as a {role}",
     "I moved from Seattle"),
]
all_pairs.extend(expand(work_templates, work_subs))

# --- 2. Location ---
loc_subs = [{"city": c, "city2": c2}
            for c, c2 in zip(CITIES[:10], CITIES[5:15])]
loc_templates = [
    ("Where does the person live?",
     "I live in {city}",
     "I work at Google"),
    ("What city is the person in?",
     "I'm based in {city}",
     "I use Python"),
    ("Where is the person located?",
     "I'm currently in {city}",
     "My birthday is March 15"),
    ("Where did the person move to?",
     "I moved to {city}",
     "I have a cat"),
    ("Where did the person live before?",
     "I used to live in {city2}",
     "I live in {city}"),
    ("What was the person's previous city?",
     "I previously lived in {city2}",
     "I currently live in {city}"),
]
all_pairs.extend(expand(loc_templates, loc_subs))

# --- 3. Programming Languages & Tools ---
lang_subs = [{"lang": l, "lang2": l2, "tool": t}
             for l, l2, t in zip(LANGUAGES[:8], LANGUAGES[4:12], TOOLS[:8])]
lang_templates = [
    ("What programming languages does the person use?",
     "I mostly code in {lang} and {lang2}",
     "I live in Berlin"),
    ("What language does the person prefer?",
     "I prefer {lang} over {lang2}",
     "My manager is Sarah"),
    ("What does the person code in?",
     "I write {lang} for work",
     "I have a dog"),
    ("What database does the team use?",
     "We use {tool} as our main database",
     "I like hiking"),
    ("What tools does the person use?",
     "I use {tool} daily",
     "My birthday is September"),
]
all_pairs.extend(expand(lang_templates, lang_subs))

# --- 4. Relationships ---
rel_subs = [{"name": n, "rel": r[0], "rel_syn": r[1], "company": c, "role": ro, "city": ci}
            for n, r, c, ro, ci in zip(
                random.sample(NAMES, 10),
                random.choices(RELATIONS, k=10),
                random.choices(COMPANIES, k=10),
                random.choices(ROLES, k=10),
                random.choices(CITIES, k=10))]
rel_templates = [
    ("Who is the person's {rel}?",
     "My {rel} is {name}",
     "I work at {company}"),
    ("Who is the person's {rel_syn}?",
     "My {rel} is {name}, works at {company}",
     "I live in {city}"),
    ("Tell me about the person's {rel}",
     "My {rel} {name} is a {role} in {city}",
     "I use Python"),
    ("Does the person have a {rel}?",
     "My {rel} is {name}",
     "I graduated from MIT"),
    ("What does the person's {rel} do?",
     "My {rel} {name} works at {company} as a {role}",
     "I like coffee"),
]
all_pairs.extend(expand(rel_templates, rel_subs))

# --- 5. Preferences & Likes ---
pref_subs = [{"food": f, "food2": f2, "hobby": h}
             for f, f2, h in zip(FOODS[:8], FOODS[2:10], HOBBIES[:8])]
pref_templates = [
    ("What does the person like to drink?",
     "I love {food}",
     "I work at Google"),
    ("What does the person enjoy?",
     "I really enjoy {hobby}",
     "I live in Berlin"),
    ("What doesn't the person like?",
     "I don't like {food2}",
     "I like {food}"),
    ("What are the person's hobbies?",
     "I'm really into {hobby}",
     "I use PostgreSQL"),
    ("What does the person do for fun?",
     "I do {hobby} on weekends",
     "My manager is Sarah"),
    ("Does the person like {food}?",
     "I love {food}",
     "I drink {food2}"),
]
all_pairs.extend(expand(pref_templates, pref_subs))

# --- 6. Negation ---
neg_subs = [{"food": f, "lang": l, "hobby": h}
            for f, l, h in zip(FOODS[:6], LANGUAGES[:6], HOBBIES[:6])]
neg_templates = [
    ("What doesn't the person like?",
     "I don't like {lang}",
     "I use {lang} every day"),
    ("What can't the person eat?",
     "I'm allergic to shellfish",
     "I like sushi"),
    ("Does the person drink {food}?",
     "I don't drink {food}",
     "I love {food}"),
    ("What does the person avoid?",
     "I can't stand {food}",
     "I prefer {food}"),
    ("Any dietary restrictions?",
     "I'm vegetarian",
     "I like steak"),
    ("Is the person allergic to anything?",
     "I'm allergic to shellfish",
     "I eat everything"),
]
all_pairs.extend(expand(neg_templates, neg_subs))

# --- 7. Education ---
edu_subs = [{"school": s, "degree": d}
            for s, d in zip(SCHOOLS[:6], DEGREES[:6])]
edu_templates = [
    ("Where did the person go to school?",
     "I graduated from {school} with a {degree} degree",
     "I work at Google"),
    ("What did the person study?",
     "I studied {degree} at {school}",
     "I live in Berlin"),
    ("What's the person's education?",
     "I have a {degree} degree from {school}",
     "I like coffee"),
    ("What university did the person attend?",
     "I went to {school}",
     "My dog is named Luna"),
]
all_pairs.extend(expand(edu_templates, edu_subs))

# --- 8. Pets ---
pet_subs = [{"pet_type": p[0], "pet_name": p[1]}
            for p in PETS]
pet_templates = [
    ("Does the person have pets?",
     "I have a {pet_type} named {pet_name}",
     "I work at Google"),
    ("What is the person's pet's name?",
     "My {pet_type} is called {pet_name}",
     "My name is Alex"),
    ("What kind of pet does the person have?",
     "I have a {pet_type} named {pet_name}",
     "I graduated from MIT"),
    ("Tell me about the person's {pet_type}",
     "My {pet_type} {pet_name} is 3 years old",
     "I drink coffee"),
]
all_pairs.extend(expand(pet_templates, pet_subs))

# --- 9. Temporal / Previous facts ---
temp_subs = [{"city": c, "city2": c2, "company": co, "company2": co2}
             for c, c2, co, co2 in zip(CITIES[:6], CITIES[6:12],
                                        COMPANIES[:6], COMPANIES[6:12])]
temp_templates = [
    ("Where did the person live before {city}?",
     "I used to live in {city2}",
     "I live in {city}"),
    ("What was the person's previous job?",
     "Before {company}, I worked at {company2}",
     "I work at {company}"),
    ("Where did the person work previously?",
     "I was at {company2} for 3 years",
     "I'm currently at {company}"),
    ("What did the person do before their current role?",
     "I was a senior engineer at {company2}",
     "I'm a tech lead at {company}"),
]
all_pairs.extend(expand(temp_templates, temp_subs))

# --- 10. Specific details ---
spec_subs = [{"device": d, "vehicle": v, "book": b}
             for d, v, b in zip(DEVICES[:5], VEHICLES[:5], BOOKS[:5])]
spec_templates = [
    ("What device does the person use?",
     "I use a {device}",
     "I work at Google"),
    ("What phone does the person have?",
     "My phone is a {device}",
     "I live in Berlin"),
    ("What car does the person drive?",
     "I drive a {vehicle}",
     "I use Python"),
    ("What is the person reading?",
     "I'm currently reading {book}",
     "I graduated from MIT"),
    ("What's the person's favorite book?",
     "My favorite book is {book}",
     "I have a dog"),
]
all_pairs.extend(expand(spec_templates, spec_subs))

# --- 11. Team & company ---
team_subs = [{"team": t, "company": c, "name": n, "role": r}
             for t, c, n, r in zip(TEAMS[:6], COMPANIES[:6],
                                    NAMES[:6], ROLES[:6])]
team_templates = [
    ("What team is the person on?",
     "I'm the tech lead for the {team} team",
     "I live in Berlin"),
    ("How big is the person's team?",
     "The {team} team has 8 engineers",
     "I work at {company}"),
    ("Who is on the person's team?",
     "I mentor {name} on the {team} team",
     "I use Python"),
    ("Who is the CTO?",
     "The CTO is {name}",
     "I graduated from MIT"),
    ("What's the company mission?",
     "{company} builds payment infrastructure",
     "I like coffee"),
]
all_pairs.extend(expand(team_templates, team_subs))

# --- 12. Birthday & personal dates ---
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
date_subs = [{"month": m, "day": str(random.randint(1, 28))}
             for m in months[:6]]
date_templates = [
    ("When is the person's birthday?",
     "My birthday is {month} {day}",
     "I work at Google"),
    ("What's the person's date of birth?",
     "I was born on {month} {day}",
     "I live in Berlin"),
]
all_pairs.extend(expand(date_templates, date_subs))

# --- 13. Casual phrasing variants ---
casual_pairs = [
    {"query": "what's my name?", "positive": "My name is Alex Thompson", "negative": "I work at Google"},
    {"query": "who am I?", "positive": "I'm Alex, a software engineer", "negative": "I live in Berlin"},
    {"query": "do I have kids?", "positive": "I don't have any kids", "negative": "I have a dog named Luna"},
    {"query": "am I married?", "positive": "Sarah and I got engaged", "negative": "I work at PayFlow"},
    {"query": "what do I drink?", "positive": "I switched from coffee to matcha", "negative": "I use Python"},
    {"query": "do I exercise?", "positive": "I run 5km every morning", "negative": "I read Clean Architecture"},
    {"query": "what's my blood type?", "positive": "My blood type is O positive", "negative": "I'm 32 years old"},
    {"query": "what event streaming do we use?", "positive": "We decided to go with Kafka", "negative": "We use PostgreSQL"},
    {"query": "what gym do I go to?", "positive": "Found a great climbing gym called Vertigo", "negative": "I live in Lisbon"},
    {"query": "do I have an espresso machine?", "positive": "Got a new espresso machine", "negative": "I switched to matcha"},
    {"query": "who left the team?", "positive": "Jake is leaving for Meta next month", "negative": "We hired two new engineers"},
    {"query": "what's our monthly volume?", "positive": "Our monthly volume hit $100M", "negative": "We handle 10k TPS"},
    {"query": "how many engineers?", "positive": "We're a Series B startup with about 80 engineers", "negative": "The payments team has 8 people"},
    {"query": "what dev process do we follow?", "positive": "We follow trunk-based development with feature flags", "negative": "We deploy to AWS"},
]
all_pairs.extend(casual_pairs)

# ============================================================
# Deduplicate and shuffle
# ============================================================

seen = set()
unique_pairs = []
for p in all_pairs:
    key = (p["query"], p["positive"])
    if key not in seen:
        seen.add(key)
        unique_pairs.append(p)

random.shuffle(unique_pairs)

# Split into train/val
split = int(len(unique_pairs) * 0.9)
train = unique_pairs[:split]
val = unique_pairs[split:]

print(f"Total unique pairs: {len(unique_pairs)}")
print(f"Train: {len(train)}, Val: {len(val)}")

# Save
output_dir = "/Users/mohammeddalaali/projects/lore-memory/scripts"
with open(f"{output_dir}/train_pairs.json", "w") as f:
    json.dump(train, f, indent=2)
with open(f"{output_dir}/val_pairs.json", "w") as f:
    json.dump(val, f, indent=2)

# Also save in sentence-transformers InputExample format
print(f"\nSample pairs:")
for p in train[:5]:
    print(f"  Q: {p['query']}")
    print(f"  +: {p['positive']}")
    print(f"  -: {p['negative']}")
    print()

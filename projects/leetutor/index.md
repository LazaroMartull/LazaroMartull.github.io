# LeetTutor – AI-Powered Programming Tutor

LeetTutor is an AI-driven tutoring system designed to provide structured, step-by-step programming guidance to post-secondary students.

This system was evaluated through real programming assignments and published as:

🔬 **LeafTutor: An AI Agent for Programming Assignment Tutoring**  
arXiv:2601.02375  
https://arxiv.org/abs/2601.02375

---

## 🧩 My Role

I contributed to:

- System architecture design  
- Backend implementation (Flask API)  
- LLM prompt structuring for step-by-step reasoning  
- Database schema design (PostgreSQL)  
- Evaluation and testing workflow  
- Documentation and research contribution  

---

## 🎯 Problem

High enrollment in STEM programs has created increasing demand for scalable programming tutoring support.

Traditional tutoring:
- Does not scale well  
- Is resource-constrained  
- Varies in instructional quality  

The goal was to design an AI system capable of delivering **clear, structured, step-by-step programming explanations comparable to human tutors.**

---

## 🧠 System Architecture

LeetTutor integrates:

- **Frontend →** Student query submission
- **Backend (Flask) →** API handling and session management
- **Database (PostgreSQL) →** Storing assignments and interactions
- **LLM Layer →** Structured prompt design for step-by-step reasoning
- **Evaluation Pipeline →** Comparing outputs against expected tutoring quality

### Core Flow

1. Student submits programming question  
2. Backend formats structured prompt  
3. LLM generates step-by-step instructional response  
4. Output is returned with organized explanation  
5. Evaluation framework measures instructional clarity and completeness  

---

## 🛠 Tools & Technologies

- Python  
- Flask  
- PostgreSQL  
- LLM API (Gemini / GPT-style models)  
- Prompt engineering  
- Research-based evaluation framework  

---

## 🧪 Prompt Engineering Strategy

To simulate tutoring behavior, prompts were structured to:

- Break problems into logical steps  
- Explain reasoning before providing solutions  
- Avoid directly dumping final answers  
- Encourage guided learning  

This structured prompting significantly improved clarity and instructional usefulness.

---

## 📊 Evaluation & Results

LeetTutor was evaluated using **real programming assignments**.

Key findings:

- Delivered structured step-by-step explanations  
- Maintained instructional clarity  
- Demonstrated guidance quality comparable to human tutors  
- Reduced ambiguity in programming explanations  

This validated the system’s potential for scalable STEM tutoring.

---

## 🔍 Research Contribution

This project contributes to:

- Applied AI in education  
- LLM tutoring systems  
- Instructional design via prompt engineering  
- Scalable academic support solutions  

The work was published on arXiv and categorized under:
- Artificial Intelligence (cs.AI)  
- Software Engineering (cs.SE)  
- Computers and Society (cs.CY)  

---

## 🚀 Future Improvements

Planned enhancements include:

- Fine-tuned LLM for tutoring-specific behavior  
- Rubric-based evaluation scoring  
- Multi-turn conversation tracking  
- Personalized learning profiles  
- Instructor dashboard analytics  

---

## 📁 Folder Structure

    projects/
    └── leetutor/
        ├── index.md              # Project write-up (this page)
        └── leaf_tutor_paper.pdf  # Research paper (PDF)

---

## 🎯 Summary

LeetTutor demonstrates how large language models can be engineered to deliver structured, pedagogically sound programming guidance.

By combining system design, backend engineering, prompt structuring, and real-world evaluation, this project bridges academic research and applied AI system development.

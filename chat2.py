import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask
from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)
app.app_context().push()
#db.init_app(app)



class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prompt = db.Column(db.Text, nullable=False) 
    response = db.Column(db.Text, nullable=False) 
    def __repr__(self):
        return f'<Conversation {self.id}>' 

def generate_text(prompt, max_length=1000):
    tokenizer = AutoTokenizer.from_pretrained(".")
    model = AutoModelForCausalLM.from_pretrained(
        ".",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

    
@app.route("/")
def index():
    conversations = Conversation.query.all()
    return render_template("index.html", Conversation=conversations)


@app.route("/add_conversation", methods=["POST"])
def add_conversation():
    prompt = request.form.get('prompt')
    
    if True:
        response = generate_text(prompt)
        print(f"Received prompt: {prompt}")
        print(f"Received prompt: {response}")
        
        new_conversation = Conversation(prompt=prompt, response=response)
        db.session.add(new_conversation)
        db.session.commit()
        return redirect(url_for("index"))
    else:
        return "Invalid input, please provide a prompt.", 400
    
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
      

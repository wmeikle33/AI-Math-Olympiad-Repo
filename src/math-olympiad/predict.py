def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    
    id_value = id_.item(0)
    question_text = question.item(0)
    
    gc.disable()
    
    final_answer = solver.solve_problem(question_text)
    
    gc.enable()
    gc.collect()
    
    return pl.DataFrame({'id': id_value, 'answer': final_answer})

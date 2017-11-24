package org.snu.cse.cap.translator.structure.task;

enum Operator {
	OPERATOR_EQUAL("=="),
	OPERATOR_GREATER(">"),
	OPERATOR_LESS("<"),
	OPERATOR_GREATER_EQUAL(">="),
	OPERATOR_LESS_EQUAL("<="),
	OPERATOR_NOT_EQUAL("!="),
	;

	private final String value;
	
	private Operator(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
}

public class Condition {
	private String leftOperand;
	private String rightOperand;
	private Operator operator;
	
	public Condition(String leftOperand, String rightOperand, String operator)
	{
		this.leftOperand = leftOperand;
		this.rightOperand = rightOperand;
		this.operator = Operator.valueOf(operator);
		
	}
	
	public String getLeftOperand() {
		return leftOperand;
	}
	
	public void setLeftOperand(String leftOperand) {
		this.leftOperand = leftOperand;
	}
	
	public String getRightOperand() {
		return rightOperand;
	}
	
	public void setRightOperand(String rightOperand) {
		this.rightOperand = rightOperand;
	}
	
	public Operator getOperator() {
		return operator;
	}
	
	public void setOperator(Operator operator) {
		this.operator = operator;
	}
}

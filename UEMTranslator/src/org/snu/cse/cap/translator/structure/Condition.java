package org.snu.cse.cap.translator.structure;

enum Operator {
	OPERATOR_EQUAL,
	OPERATOR_GREATER,
	OPERATOR_LESS,
	OPERATOR_GREATER_EQUAL,
	OPERATOR_LESS_EQUAL,
	OPERATOR_NOT_EQUAL,
}

public class Condition {
	private String leftOperand;
	private String rightOperand;
	private Operator operator;
	
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

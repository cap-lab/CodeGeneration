package org.snu.cse.cap.translator.structure.task;

import org.snu.cse.cap.translator.structure.device.ArchitectureType;

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
	
	public static Operator fromValue(String value) {
		 for (Operator c : Operator.values()) {
			 if (c.value.equals(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
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
		this.operator = Operator.fromValue(operator);
		
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


package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for TaskGraphPropertyType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="TaskGraphPropertyType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="ProcessNetwork"/&gt;
 *     &lt;enumeration value="DataFlow"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "TaskGraphPropertyType")
@XmlEnum
public enum TaskGraphPropertyType {

    @XmlEnumValue("ProcessNetwork")
    PROCESS_NETWORK("ProcessNetwork"),
    @XmlEnumValue("DataFlow")
    DATA_FLOW("DataFlow");
    private final String value;

    TaskGraphPropertyType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static TaskGraphPropertyType fromValue(String v) {
        for (TaskGraphPropertyType c: TaskGraphPropertyType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}

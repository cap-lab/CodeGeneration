
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ArchitectureElementCategoryType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="ArchitectureElementCategoryType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="processor"/&gt;
 *     &lt;enumeration value="memory"/&gt;
 *     &lt;enumeration value="dma"/&gt;
 *     &lt;enumeration value="hwip"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "ArchitectureElementCategoryType")
@XmlEnum
public enum ArchitectureElementCategoryType {

    @XmlEnumValue("processor")
    PROCESSOR("processor"),
    @XmlEnumValue("memory")
    MEMORY("memory"),
    @XmlEnumValue("dma")
    DMA("dma"),
    @XmlEnumValue("hwip")
    HWIP("hwip");
    private final String value;

    ArchitectureElementCategoryType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static ArchitectureElementCategoryType fromValue(String v) {
        for (ArchitectureElementCategoryType c: ArchitectureElementCategoryType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}

import {
  Component,
  ElementRef,
  inject,
  OnInit,
  ViewChild,
} from '@angular/core';
import {
  AbstractControl,
  FormBuilder,
  FormGroup,
  ReactiveFormsModule,
  ValidatorFn,
  Validators,
} from '@angular/forms';
import { TextInputComponent } from '../_components/text-input/text-input.component';
import { MessageBoxComponent } from '../_components/message-box/message-box.component';
import { Message } from '../_models/message';
import { SelectInputComponent } from '../_components/select-input/select-input.component';
import { ChatbotService } from '../_services/chatbot.service';
import { Prompt } from '../_models/prompt';

@Component({
  selector: 'app-manual-entries',
  standalone: true,
  imports: [
    ReactiveFormsModule,
    TextInputComponent,
    MessageBoxComponent,
    SelectInputComponent,
  ],
  templateUrl: './manual-entries.component.html',
  styleUrls: ['./manual-entries.component.css'],
})
export class ManualEntriesComponent implements OnInit {
  @ViewChild('form') form: any;
  @ViewChild('conversationContainer')
  private conversationContainer = inject(ElementRef);
  private fb = inject(FormBuilder);
  private chatbotService = inject(ChatbotService);

  clientForm: FormGroup = new FormGroup({});
  conversation: Message[] = [
    {
      sender: 'user',
      message: '',
    },
    {
      sender: 'bot',
      message: '',
    },
  ];
  riskRates: string[] = ['Not Assigned', 'Low', 'Medium', 'High'];
  avgOrderRates: string[] = ['Constant', 'Decreased', 'Increased'];
  avgQuantityOrderedRates: string[] = ['Constant', 'Decreased', 'Increased'];
  completedOrdersRatios: string[] = [
    'All',
    'More Than Half',
    'Less Than Half',
    'None',
  ];
  canceledOrdersRatios: string[] = [
    'All',
    'Most',
    'Moderate',
    'Little',
    'None',
  ];
  orderTypes: string[] = ['Sell', 'Buy'];
  executionStatus: string[] = [
    'Executed',
    'Partially Executed',
    'Not Executed',
  ];
  sectorNames: string[] = [
    'Industries',
    'Financials',
    'Real Estate',
    'Materials',
    'Energy',
    'INVESTMENT',
    'Consumer Discretionary',
    'INDUSTRIAL',
    'Information Technology',
    'Health Care',
    'Consumer Staples',
    'REAL ESTATE',
    'Telecommunication Services',
    'Basic Materials',
    'Others',
    'FOOD',
    'Tourism',
    'Telecommunications',
    'SERVICES',
  ];

  ngOnInit(): void {
    this.initializeForm();
    this.updateConversationMessage(this.clientForm.value);

    this.clientForm.valueChanges.subscribe((formValue) => {
      this.updateConversationMessage(formValue);
    });
  }

  initializeForm() {
    this.clientForm = this.fb.group({
      gender: ['Male'],
      age: [
        40,
        [
          Validators.required,
          Validators.min(1),
          Validators.max(100),
          this.integerValidator(),
        ],
      ],
      riskRate: ['Not Assigned'],
      avgPrice: [9.56, [Validators.required, Validators.min(0)]],
      avgOrderRateDiff: ['Constant'],
      avgQuantityOrderRateDiff: ['Constant'],
      completedOrdersRatio: ['More Than Half'],
      canceledOrdersRatio: ['Moderate'],
      orderType: ['Sell'],
      executionStatus: ['Executed'],
      sectorName: ['Financials'],
    });
  }

  integerValidator(): ValidatorFn {
    return (control: AbstractControl) => {
      return Number.isInteger(Number(control.value))
        ? null
        : { notInteger: true };
    };
  }

  updateConversationMessage(formValue: any) {
    this.conversation[0].message = `The client is a ${formValue['gender']}. The client is ${formValue['age']} years old. The client's risk rate is ${formValue['riskRate']}. The client's average price of orders is ${formValue['avgPrice']}. The client's average order rate is ${formValue['avgOrderRateDiff']}. The client's average quantity ordered rate is ${formValue['avgQuantityOrderRateDiff']}. The client's completed orders ratio is ${formValue['completedOrdersRatio']}. the client's canceled orders ratio is ${formValue['canceledOrdersRatio']}. The client's most frequent order type is ${formValue['orderType']}. The client's most frequent execution status is ${formValue['executionStatus']}. The client's most frequent sector name is ${formValue['sectorName']}. Will they churn?`;
  }

  submitForm() {
    if (this.form) {
      this.form.ngSubmit.emit();
    }
  }

  predict() {
    this.conversation[1].message = '';
    const prompt: Prompt = {
      prompt: this.conversation[0].message,
      isSinglePrompt: true,
    };
    console.log(prompt);
    this.chatbotService.send_prompt(prompt).subscribe({
      next: (response) => {
        this.conversation[1] = { sender: 'bot', message: response.response };
        setTimeout(() => this.scrollToBottom(), 0);
      },
    });
    setTimeout(() => this.scrollToBottom(), 0);
  }

  scrollToBottom(): void {
    try {
      const container = this.conversationContainer.nativeElement;
      container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth',
      });
    } catch (err) {
      console.log('Scroll to bottom failed:', err);
    }
  }
}
